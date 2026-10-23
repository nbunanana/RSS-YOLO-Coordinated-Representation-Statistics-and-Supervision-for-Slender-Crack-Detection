"""seg → det 权重键转移 + box 头键覆盖断言 + 双路径一致性核对。

用途
----
掩膜辅助监督臂的 box 指标必须与 detect 臂**同口径**。本脚本把 seg 臂 `best.pt` 中与
detect yaml **同名**的键转移到 detect 结构上：
  * **断言 box head 键 100% 匹配**（打印 matched/total）；不对 100% 就报错退出；
  * 落盘 `seg2det.pt`，供后续分析脚本复用（用 --emit-pseudo-run）；
  * 跑 val/test 的 box 指标（Detect/DetectionValidator 口径）；
  * 双路径核对（在任一 run 上额外用"直接对 seg 模型 `model.val()` 取 `.box`"），要求与转移路径
    的 box 指标差 ≤ 0.1 点，**并逐个 run 目录报配对值**；超出容差时在输出中标记提示。

用法
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
    """定位 ultralytics 包根目录：环境变量 > 源码树 > 已安装包。"""
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
RUNS = Path(os.environ.get("CRACK_SEG_RUNS", UM / "runs" / "segment"))   # seg 训练产物根目录
PSEUDO = Path(os.environ.get("CRACK_DET_RUNS", UM / "runs" / "detect"))   # seg→det 伪 run 的输出根目录
DATA = "datasets/crack-seg-712.yaml"
ZS_DATA = "ultralytics/cfg/datasets/robotflow_crack.yaml"
TOL_POINT = 0.1  # 双路径一致性容差（点）


def _rel(p: Path) -> str:
    """尽量返回相对 REPO 的路径，超出 REPO 时退化为绝对路径。"""
    try:
        return str(Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="seg run 目录（相对 runs 根目录的路径）")
    ap.add_argument("--det-yaml", default=str(Path(__file__).resolve().parent.parent / "configs" / "yolov8-ema-crack-nodup.yaml"),
                    help="与 seg 臂 Neck 逐层一致的 detect yaml（默认取本包 configs/ 下的部署形态配置）")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--data", default=DATA,
                    help="数据集 yaml（默认主集 crack-seg-712；第二数据集用 --data datasets/crackseg9k.yaml）")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--dual-path", action="store_true", help="额外跑 seg 模型自身的 .box（一致性核对）")
    ap.add_argument("--zeroshot", action="store_true", help="对 seg2det.pt 跑跨域零样本")
    ap.add_argument("--emit-pseudo-run", dest="emit_pseudo", action="store_true",
                    help="把 seg2det.pt 放到 runs/detect/<project>/<派生名>/weights/best.pt，"
                         "使下游分析可直接复用（派生名由 run 名加 2det 标记构成）")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if UM.is_dir():
        sys.path.insert(0, str(UM))
        os.chdir(UM)
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import intersect_dicts

    # 关闭 pin_memory 并限制 prefetch：只影响吞吐，不影响指标（部分机器上可避免 pin-memory 报错）
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
            print(f"[跳过] {run}: 找不到 {seg_w}")
            continue
        print("=" * 100)
        print(f"{run}   seg 权重 = {_rel(seg_w)}")
        seg = YOLO(str(seg_w))
        det = YOLO(args.det_yaml, task="detect")
        sd = det.model.state_dict()
        csd = seg.model.float().state_dict()
        updated = intersect_dicts(csd, sd, exclude=("proto", "cv4"))  # 只取 detect 结构里存在的键
        n_head = len(det.model.model)
        head_keys = [k for k in sd if k.startswith(f"model.{n_head - 1}.")]
        head_hit = [k for k in head_keys if k in updated]
        n_hit, n_all = len(updated), len(sd)
        conv_hit = sum(1 for k in updated if k.endswith(".conv.weight"))
        conv_all = sum(1 for k in sd if k.endswith(".conv.weight"))
        print(f"  Transferred {n_hit}/{n_all} items（含卷积层 {conv_hit}/{conv_all}）")
        print(f"  box head 键覆盖：{len(head_hit)}/{len(head_keys)}  "
              f"{'OK ✓（100% 匹配）' if len(head_hit) == len(head_keys) else '*** 未 100% 匹配，终止 ***'}")
        if len(head_hit) != len(head_keys):
            missing = sorted(set(head_keys) - set(head_hit))
            print(f"  缺失示例：{missing[:8]}")
            return 2
        missing_all = sorted(set(sd) - set(updated))
        if missing_all:
            # 硬失败：det-yaml 选错时会静默留下随机初始化的分支（box head 断言无法发现，因为两 yaml 的
            # box head 键相同）。对"单变量键转移"而言，exclude(proto/cv4) 之外不应有任何未转移键。
            print(f"  [错误] 仍有 {len(missing_all)} 个键未转移（非 box head）：{missing_all[:8]}")
            print("  ⇒ det-yaml 很可能不是该 seg 结构的 detect 孪生；终止以避免用随机初始化分支出指标。")
            return 2
        det.model.load_state_dict(updated, strict=False)

        # 落盘 seg2det.pt（Model.save 需要 ckpt 为 dict）
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
            print(f"  -> 伪 run 目录 {_rel(pdir)}（供下游分析复用）")

        rec = {"run": run, "transferred": n_hit, "total": n_all,
               "head_matched": len(head_hit), "head_total": len(head_keys)}
        for split in args.splits:
            vd = det.val(data=args.data, split=split, imgsz=args.imgsz, batch=args.batch,
                         device=args.device, plots=False, verbose=False)
            rec[f"{split}_det_map50"] = round(float(vd.box.map50), 4)
            rec[f"{split}_det_map"] = round(float(vd.box.map), 4)
            rec[f"{split}_det_P"] = round(float(vd.box.mp), 4)
            rec[f"{split}_det_R"] = round(float(vd.box.mr), 4)
            line1 = (f"  [{split}] 转移路径(detect validator)  P={rec[f'{split}_det_P']:.4f} "
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
                line1 += (f"\n            双路径(seg 模型自身 .box) mAP50={s50:.4f} mAP50-95={s95:.4f}"
                          f"  Δ={d50:.3f}/{d95:.3f} 点  "
                          f"{'OK ✓' if max(d50, d95) <= TOL_POINT else '*** 超出 0.1 点容差，请核对 ***'}")
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
