# -*- coding: utf-8 -*-
"""自检：构建论文的三条实验臂，核对参数量与一次前向。

用法（在 code/ 目录下，且 overlay 已覆盖到 ultralytics）：
    python verify.py

期望输出三个参数量：3,011,715 / 3,264,483 / 3,264,483。
若与本目录 README 第 8 节不一致，说明 overlay 未正确覆盖或版本不符。
"""
import sys
from pathlib import Path

import torch

CFG = Path(__file__).resolve().parent / "configs"
EXPECT = {
    "yolov8-ema-crack-nodup.yaml": 3_011_715,      # 部署形态：Detect 头
    "yolov8-ema-crack-nodup-seg.yaml": 3_264_483,  # 训练形态：Segment 头 + 掩膜分支
    "yolov8-ema-only-seg.yaml": 3_264_483,         # 去 GN 对照臂
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
        print(f"  [{flag}] {name:36s} params={params:,} (期望 {expect:,}) forward={shape}")

    # 关键构件可用性
    from ultralytics.nn.modules import C2f_EMA, C2f_GN
    from ultralytics.utils.loss import v8SegmentationLoss  # noqa: F401

    print("  C2f_EMA / C2f_GN / v8SegmentationLoss 导入正常")
    print("\n全部通过" if ok else "\n存在不一致，请检查 overlay 覆盖路径与 ultralytics 版本")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
