"""seg to det weight-key transfer, box-head key coverage assertion, and dual-path consistency check.

Purpose
-------
The box metrics of the mask-auxiliary supervised configuration must follow the same protocol as the
detect configuration. This script transfers the same-named keys of the seg `best.pt` onto the detect yaml:
  * assert a 100 per cent match of the box head keys (print matched/total); exit with an error otherwise;
  * write `seg2det.pt` for reuse by later analysis scripts (with --emit-pseudo-run);
  * run the box metrics of val/test (Detect/DetectionValidator protocol);
  * dual-path check (on any run, additionally read `.box` from `model.val()` of the seg model), requiring
    the box metrics to stay within 0.1 point of the transfer path; report paired values and flag exceedances.

Usage
    python seg2det_eval.py --runs <RUN_DIR> --splits val test --dual-path
    python seg2det_eval.py --runs <RUN_DIR> [<RUN_DIR> ...] --emit-pseudo-run
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _resolve_um() -> Path:
    """Locate the ultralytics package root: environment variable first, then source tree, then installed package."""
    env = os.environ.get("CRACK_ULTRALYTICS_ROOT")
    if env:
        return Path(env)
    local = REPO / "ultralytics-main"
    if (local / "ultralytics").is_dir():
        return local
    try:
        import importlib.util

        spec = importlib.util.find_spec("ultralytics")
        if spec and spec.origin:
            return Path(spec.origin).resolve().parent.parent
    except Exception:
        pass
    return local


UM = _resolve_um()
RUNS = Path(os.environ.get("CRACK_SEG_RUNS", UM / "runs" / "segment"))   # root of the seg training artefacts
PSEUDO = Path(os.environ.get("CRACK_DET_RUNS", UM / "runs" / "detect"))   # output root for seg to det pseudo runs
DATA = "datasets/crack-seg-712.yaml"
ZS_DATA = "ultralytics/cfg/datasets/robotflow_crack.yaml"
TOL_POINT = 0.1  # dual-path consistency tolerance (points)


def _rel(p: Path) -> str:
    """Return the path relative to REPO where possible, falling back to the absolute path outside REPO."""
    try:
        return str(Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="seg run directory (path relative to the runs root)")
    ap.add_argument("--det-yaml", default=str(Path(__file__).resolve().parent.parent / "configs" / "yolov8-ema-crack-nodup.yaml"),
                    help="detect yaml whose Neck matches the seg configuration layer by layer (defaults to the deployment form under configs/)")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--data", default=DATA,
                    help="dataset yaml (defaults to the primary set crack-seg-712; use --data datasets/crackseg9k.yaml for the second set)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--dual-path", action="store_true", help="additionally run .box of the seg model itself (consistency check)")
    ap.add_argument("--zeroshot", action="store_true", help="run the cross-domain zero-shot evaluation on seg2det.pt")
    ap.add_argument("--emit-pseudo-run", dest="emit_pseudo", action="store_true",
                    help="place seg2det.pt at runs/detect/<project>/<derived name>/weights/best.pt so that "
                         "downstream analysis can reuse it directly (the derived name is the run name plus a 2det marker)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if UM.is_dir():
        sys.path.insert(0, str(UM))
        os.chdir(UM)
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import intersect_dicts

    # disable pin_memory and limit prefetch: this changes throughput only, not the metrics (avoids pin-memory errors on some machines)
    import torch.utils.data as _tud
    _orig_dl_init = _tud.DataLoader.__init__

    def _dl_init(self, *a, **kw):
        kw["pin_memory"] = False
        if kw.get("num_workers", 0) and kw.get("prefetch_factor") is not None:
            kw["prefetch_factor"] = 2
        _orig_dl_init(self, *a, **kw)

    _tud.DataLoader.__init__ = _dl_init

    rows = []
    for run in args.runs:
        seg_w = RUNS / run / "weights" / "best.pt"
        if not seg_w.exists():
            print(f"[skip] {run}: cannot find {seg_w}")
            continue
        print("=" * 100)
        print(f"{run}   seg weights = {_rel(seg_w)}")
        seg = YOLO(str(seg_w))
        det = YOLO(args.det_yaml, task="detect")
        sd = det.model.state_dict()
        csd = seg.model.float().state_dict()
        updated = intersect_dicts(csd, sd, exclude=("proto", "cv4"))  # keep only the keys that exist in the detect structure
        n_head = len(det.model.model)
        head_keys = [k for k in sd if k.startswith(f"model.{n_head - 1}.")]
        head_hit = [k for k in head_keys if k in updated]
        n_hit, n_all = len(updated), len(sd)
        conv_hit = sum(1 for k in updated if k.endswith(".conv.weight"))
        conv_all = sum(1 for k in sd if k.endswith(".conv.weight"))
        print(f"  Transferred {n_hit}/{n_all} items (conv layers {conv_hit}/{conv_all})")
        print(f"  box head key coverage: {len(head_hit)}/{len(head_keys)}  "
              f"{'OK ✓ (100 per cent match)' if len(head_hit) == len(head_keys) else '*** not a 100 per cent match, aborting ***'}")
        if len(head_hit) != len(head_keys):
            missing = sorted(set(head_keys) - set(head_hit))
            print(f"  missing examples: {missing[:8]}")
            return 2
        missing_all = sorted(set(sd) - set(updated))
        if missing_all:
            # hard failure: a wrong det-yaml silently leaves randomly initialised branches that the box
            # head assertion cannot detect; for a transfer of this kind no key outside exclude should remain.
            print(f"  [error] {len(missing_all)} keys are still untransferred (non box head): {missing_all[:8]}")
            print("  => det-yaml is probably not the detect twin of this seg structure; aborting to avoid reporting metrics from randomly initialised branches.")
            return 2
        det.model.load_state_dict(updated, strict=False)

        # write seg2det.pt (Model.save requires ckpt to be a dict)
        out_pt = RUNS / run / "weights" / "seg2det.pt"
        det.ckpt = {}
        det.save(str(out_pt))
        print(f"  -> {_rel(out_pt)}")
        if args.emit_pseudo:
            m = re.match(r"^(.*)_([^_]+)$", Path(run).name)
            base, suffix = (m.group(1) + "2det", m.group(2)) if m else (Path(run).name + "2det", "x")
            pdir = PSEUDO / run.split("/")[0] / f"{base}_{suffix}"
            (pdir / "weights").mkdir(parents=True, exist_ok=True)
            det.save(str(pdir / "weights" / "best.pt"))
            (pdir / ".DONE").touch()
            print(f"  -> pseudo run directory {_rel(pdir)} (for downstream analysis)")

        rec = {"run": run, "transferred": n_hit, "total": n_all,
               "head_matched": len(head_hit), "head_total": len(head_keys)}
        for split in args.splits:
            vd = det.val(data=args.data, split=split, imgsz=args.imgsz, batch=args.batch,
                         device=args.device, plots=False, verbose=False)
            rec[f"{split}_det_map50"] = round(float(vd.box.map50), 4)
            rec[f"{split}_det_map"] = round(float(vd.box.map), 4)
            rec[f"{split}_det_P"] = round(float(vd.box.mp), 4)
            rec[f"{split}_det_R"] = round(float(vd.box.mr), 4)
            line1 = (f"  [{split}] transfer path (detect validator)  P={rec[f'{split}_det_P']:.4f} "
                     f"R={rec[f'{split}_det_R']:.4f} mAP50={rec[f'{split}_det_map50']:.4f} "
                     f"mAP50-95={rec[f'{split}_det_map']:.4f}")
            if args.dual_path:
                vs = YOLO(str(seg_w)).val(data=args.data, split=split, imgsz=args.imgsz, batch=args.batch,
                                          device=args.device, plots=False, verbose=False)
                s50, s95 = float(vs.box.map50), float(vs.box.map)
                d50 = abs(s50 - rec[f"{split}_det_map50"]) * 100
                d95 = abs(s95 - rec[f"{split}_det_map"]) * 100
                rec[f"{split}_seg_map50"], rec[f"{split}_seg_map"] = round(s50, 4), round(s95, 4)
                rec[f"{split}_d50_pts"], rec[f"{split}_d95_pts"] = round(d50, 3), round(d95, 3)
                line1 += (f"\n            dual path (seg model .box) mAP50={s50:.4f} mAP50-95={s95:.4f}"
                          f"  delta={d50:.3f}/{d95:.3f} points  "
                          f"{'OK ✓' if max(d50, d95) <= TOL_POINT else '*** beyond the 0.1 point tolerance, please check ***'}")
            print(line1)
        if args.zeroshot:
            vd = det.val(data=ZS_DATA, split="test", imgsz=args.imgsz, batch=args.batch,
                         device=args.device, plots=False, verbose=False)
            rec["zs_map50"], rec["zs_map"] = round(float(vd.box.map50), 4), round(float(vd.box.map), 4)
            print(f"  [zeroshot] mAP50={rec['zs_map50']:.4f} mAP50-95={rec['zs_map']:.4f}")
        rows.append(rec)

    if not rows:
        return 1
    out = PSEUDO / args.runs[0].split("/")[0] / "_summary" / f"seg2det{args.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [k for r in rows for k in r]
    cols = list(dict.fromkeys(cols))
    with open(out, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, restval="")
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n-> {_rel(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
