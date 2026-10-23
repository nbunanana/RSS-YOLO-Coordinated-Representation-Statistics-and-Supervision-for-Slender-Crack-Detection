"""重新划分 Crack-Seg 数据集为 7:1:2（按比例取整），固定随机种子。

- 原始划分**保持不动**，便于回滚与对照；
- 新建 datasets/crack-seg-712/，把全部图像与标注重新分配后**复制**进去；
- 同时生成 datasets/crack-seg-712.yaml 与 split_manifest.csv（记录每张图属于哪个 split，保证可复现）。

用法：
    python resplit_dataset.py
    python resplit_dataset.py --ratios 0.7 0.1 0.2
"""
import argparse
import csv
import os
import random
import shutil
from pathlib import Path

DS = Path(os.environ.get("CRACK_DATA_ROOT", "D:/yolov8/ultralytics-main/datasets"))
SRC = DS / "crack-seg"
DST = DS / "crack-seg-712"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect():
    items = []
    for split in ("train", "val", "test"):
        idir = SRC / "images" / split
        if not idir.exists():
            raise SystemExit(f"缺少目录: {idir}")
        for img in sorted(idir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            items.append((img, SRC / "labels" / split / f"{img.stem}.txt", split))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.1, 0.2])
    args = ap.parse_args()

    if DST.exists():
        raise SystemExit(f"{DST} 已存在，请先删除或改名，避免新旧混淆")

    items = collect()
    n = len(items)

    # 校验：标注齐全、stem 不重复
    missing = [str(i[0]) for i in items if not i[1].exists()]
    if missing:
        raise SystemExit(f"有 {len(missing)} 张图缺标注，例如: {missing[0]}")
    stems = [i[0].stem for i in items]
    if len(set(stems)) != n:
        raise SystemExit("存在重复的图片文件名（stem），会破坏图-标注配对")

    n_tr = round(n * args.ratios[0])
    n_va = round(n * args.ratios[1])
    n_te = n - n_tr - n_va

    random.Random(args.seed).shuffle(items)
    assign = (
        [("train", it) for it in items[:n_tr]]
        + [("val", it) for it in items[n_tr : n_tr + n_va]]
        + [("test", it) for it in items[n_tr + n_va :]]
    )

    for split in ("train", "val", "test"):
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, (img, lab, _orig) in assign:
        shutil.copy2(img, DST / "images" / split / img.name)
        shutil.copy2(lab, DST / "labels" / split / lab.name)

    (DS / "crack-seg-712.yaml").write_text(
        "# Crack-Seg 重新划分（7:1:2），由 scripts/resplit_dataset.py 生成\n"
        f"# seed={args.seed}, ratios={args.ratios}\n"
        "path: crack-seg-712\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n  0: crack\n",
        encoding="utf-8",
    )

    with open(DST / "split_manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "split"])
        for split, (img, _lab, _orig) in assign:
            w.writerow([img.stem, split])

    print("=" * 60)
    orig_counts = {s: sum(1 for it in items if it[2] == s) for s in ("train", "val", "test")}
    print(f"原始划分 : train={orig_counts['train']}  val={orig_counts['val']}  test={orig_counts['test']}")
    print(f"总计     : {n}")
    print(f"新划分   : train={n_tr}  val={n_va}  test={n_te}   (seed={args.seed})")
    for split in ("train", "val", "test"):
        c = len(list((DST / "images" / split).iterdir()))
        cl = len(list((DST / "labels" / split).iterdir()))
        print(f"  images/{split} = {c}   labels/{split} = {cl}")
    print(f"数据集目录: {DST}")
    print(f"yaml      : {DS / 'crack-seg-712.yaml'}")
    print(f"清单      : {DST / 'split_manifest.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
