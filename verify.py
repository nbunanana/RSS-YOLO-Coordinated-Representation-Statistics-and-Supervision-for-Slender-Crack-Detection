# -*- coding: utf-8 -*-
"""Self-check: build the configurations used in the paper and verify parameter counts and one forward pass.

Usage (from the code/ directory, with the overlay already applied to ultralytics):
    python verify.py

Expected parameter counts: 3,011,715 / 3,264,483 / 3,264,483.
A mismatch with Section 8 of the README in this directory means the overlay is not applied correctly or the versions differ.
"""
import sys
from pathlib import Path

import torch

CFG = Path(__file__).resolve().parent / "configs"
EXPECT = {
    "yolov8-ema-crack-nodup.yaml": 3_011_715,      # deployment form: Detect head
    "yolov8-ema-crack-nodup-seg.yaml": 3_264_483,  # training form: Segment head with mask branch
    "yolov8-ema-only-seg.yaml": 3_264_483,         # control configuration without GN
}


def main() -> int:
    import ultralytics
    from ultralytics import YOLO

    print(f"ultralytics {ultralytics.__version__} | torch {torch.__version__}")
    ok = True
    for name, expect in EXPECT.items():
        model = YOLO(str(CFG / name)).model
        params = sum(p.numel() for p in model.parameters())
        out = model(torch.zeros(1, 3, 640, 640, dtype=torch.float32))
        head = out[0] if isinstance(out, (list, tuple)) else out
        shape = tuple(head["boxes"].shape) if isinstance(head, dict) else "-"
        flag = "OK " if params == expect else "FAIL"
        if params != expect:
            ok = False
        print(f"  [{flag}] {name:36s} params={params:,} (expected {expect:,}) forward={shape}")

    # availability of the key building blocks
    from ultralytics.nn.modules import C2f_EMA, C2f_GN
    from ultralytics.utils.loss import v8SegmentationLoss  # noqa: F401

    print("  C2f_EMA / C2f_GN / v8SegmentationLoss imported successfully")
    print("\nall checks passed" if ok else "\ninconsistency found, check the overlay path and the ultralytics version")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
