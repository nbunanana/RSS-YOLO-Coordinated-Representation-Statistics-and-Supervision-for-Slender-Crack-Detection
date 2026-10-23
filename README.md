# RSS-YOLO 核心代码

本目录收录用于得出论文《RSS-YOLO: Coordinated Representation, Statistics and Supervision for Slender Crack Detection》全部数值结论的代码，按“模型结构 → 损失 → 训练 → 评估”四段组织。

不包含任何数据：无数据集、无图像与标注、无预训练权重、无训练日志与结果目录。仅含逻辑实现与配置。

---

## 1. 与论文的对应关系

论文采用三条实验臂，全部在 `yolov8n` 基线上做加法，配置位于 `configs/`：

| 论文中的表述 | 配置文件 | 结构差异 | 参数量（实测） |
|---|---|---|---|
| 部署形态（检测结构，用于表 2、表 3 与全部 mAP 数值） | `yolov8-ema-crack-nodup.yaml` | P3→`C2f_EMA`，P4/P5 颈部→`C2f_GN`，`Detect` 头 | 3,011,715 |
| 训练形态（含训练期掩膜分支） | `yolov8-ema-crack-nodup-seg.yaml` | 同上，末层改为 `Segment`，proto 取自 P3 | 3,264,483 |
| 去 GN 对照臂（4.1 节 `+EMA+Mask` 一类对照） | `yolov8-ema-only-seg.yaml` | 层 12/18/21 的 `C2f_GN` 换回标准 `C2f` | 3,264,483 |

两处参数量与论文 4.1 节、5.1 节报告的数值逐位一致，可作为环境是否正确的最快判据（见 `verify.py`）。

对应关系：`C2f_GN` 与 `C2f` 参数量相同（GN 与 BN 均为逐通道仿射，参数中性），因此“去 GN 臂”与本配方容量完全隔离，两臂差值只能归于机制而非参数量——这正是论文 2.3 节的论证基础。

---

## 2. 目录结构

```
code/
├── overlay/ultralytics/          # 覆盖到 ultralytics 包对应路径的改动文件
│   ├── nn/modules/attention.py   # EMA 多尺度注意力
│   ├── nn/modules/block.py       # C2f_EMA、C2f_GN
│   ├── nn/modules/conv.py        # 含 Conv_GN / Bottleneck_GN
│   ├── nn/modules/head.py        # Segment 头：proto 取自 P3（CRACK_PROTO_LVL）
│   ├── nn/modules/__init__.py    # 导出
│   ├── nn/tasks.py               # parse_model 注册
│   └── utils/loss.py             # CRACK_SEG_W / CRACK_SEG_SHUFFLE
├── configs/                      # 三条臂的模型配置
├── scripts/
│   ├── resplit_dataset.py        # Crack-Seg 重划为 7:1:2（划分种子由 --seed 指定）
│   └── seg2det_eval.py           # seg→det 权重键转移 + box 头 100% 匹配断言 + 双路径核对
├── verify.py                     # 自检：构建三条臂并核对参数量
└── requirements.txt
```

---

## 3. 关键实现

### 3.1 表征层：`C2f_EMA`（`nn/modules/block.py`）

`C2f_EMA` 继承 `C2f`，只在 `cv2` 融合之后追加一次 `EMA` 跨空间重标定：

```python
class C2f_EMA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.attn = EMA(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.attn(self.cv2(torch.cat(y, 1)))
```

通道流与 Bottleneck 数量均不变，插入位置由配置决定（论文只在层 15，即 P3）。EMA 内部为通道分组后按组共享权重、1×k 与 k×1 两路整行整列池化，`factor=8`，参数 672 个。

### 3.2 统计层：`C2f_GN`（`nn/modules/block.py`）

`C2f_GN` 与 `C2f` 逐层同构，仅把 `Conv` 换成 `Conv_GN`、`Bottleneck` 换成 `Bottleneck_GN`，即把归一化从批维换成通道分组内统计。参数个数与 BN 版完全相同。

### 3.3 监督层：掩膜辅助监督（`utils/loss.py`）

两处实现，是论文 2.4 节“可单独断开监督项”的关键：

```python
CRACK_SEG_W = float(os.environ.get("CRACK_SEG_W", "1.0"))
...
loss[4] *= self.hyp.box * CRACK_SEG_W   # 语义项（本数据集无标签，不生效）
loss[1] *= self.hyp.box * CRACK_SEG_W   # 掩膜项
```

上游实现里掩膜项与框回归共用同一个 `hyp.box`，若用 `hyp.box=0` 关掩膜会连框回归一起清零，就不是“仅断开掩膜”的对照。因此单独引入只作用于掩膜项的系数：

- `CRACK_SEG_W=1.0`（默认）：与原版逐位一致，正式臂；
- `CRACK_SEG_W=0`：仅断开掩膜监督，框/分类/DFL 监督不变。

错位对照（论文 4.2 节）由 `CRACK_SEG_SHUFFLE=1` 触发，在同一 batch 内把“预测掩膜 ↔ 真值实例”的对应按样本维置换且不含不动点，其余设置与正式臂完全一致。

---

## 4. 环境与安装

依赖见 `requirements.txt`。ultralytics 版本为 **8.4.81**，`overlay/` 中的文件按相对路径覆盖即可：

```bash
pip install -r requirements.txt
# 覆盖改动文件（示例：Linux/macOS 的 venv）
cp -r overlay/ultralytics/* "$(python -c 'import ultralytics,os;print(os.path.dirname(ultralytics.__file__))')"/
python verify.py
```

`verify.py` 会构建三条臂并断言参数量为 3,011,715 / 3,264,483 / 3,264,483；与本目录给出一致即说明覆盖成功。

---

## 4.5 种子与实验规模

本代码包不附带任何 run 清单、种子列表或结果文件，也不含任何预置的训练种子默认值；种子由使用者在训练命令中自行指定（`seed=` 参数）。因此包内不存在可据以推断具体种子编号或实验总量统计的硬编码、枚举或注释。

---

## 5. 关键环境变量

`CRACK_*` 系列是环境变量，**不会写进 `args.yaml`**，必须在启动训练/评估的同一进程内显式注入，否则结果不可比：

| 变量 | 论文取值 | 作用 |
|---|---|---|
| `CRACK_IOU` | `ciou` | 框回归损失类型。代码默认 `siou`，**论文协议为 `ciou`，必须显式覆盖** |
| `CRACK_NORM` | `bn` | 主干归一化方式，论文保持 BN |
| `CRACK_PROTO_LVL` | `0` | proto 取自 P3（`x[0]`） |
| `CRACK_SEG_W` | `1.0` / `0` | 掩膜项系数：1 为正式臂，0 为无掩膜对照 |
| `CRACK_SEG_SHUFFLE` | `0` / `1` | 1 为错位对照臂（论文 4.2 节） |
| `CRACK_SEG_SHUFFLE_SEED` | `0` | 错位对照样本置换的序列种子（仅作用于置换，与训练随机种子无关，固定值以保证逐次可复现） |

训练超参（`epochs=300`、`imgsz=640`、`batch=16`、`mosaic=0.0`、`deterministic=true`、`pretrained=yolov8n.pt`、`freeze=null`）见第 6 节训练命令。

---

## 6. 复现步骤

```bash
# 1) 重划数据为 7:1:2（划分种子须固定为一值，才能复现同一划分），生成 datasets/crack-seg-712 与 split_manifest.csv
python scripts/resplit_dataset.py --seed <SEED>

# 2) 训练（示例：正式臂）
CRACK_IOU=ciou CRACK_NORM=bn CRACK_PROTO_LVL=0 CRACK_SEG_W=1.0 \
python -c "from ultralytics import YOLO; YOLO('configs/yolov8-ema-crack-nodup-seg.yaml').train(data='crack-seg-712.yaml', epochs=300, imgsz=640, batch=16, seed=<SEED>, mosaic=0.0, deterministic=True, pretrained='yolov8n.pt')"

# 3) 同口径评估：seg 权重键转移到 det 结构，断言 box 头 100% 匹配，再报 val/test mAP
python scripts/seg2det_eval.py --runs <RUN_DIR> --det-yaml configs/yolov8-ema-crack-nodup.yaml --splits val test --dual-path

# 4) 每个 run 重复第 3 步；seg2det_eval 会把 val/test 的 box 指标写入各自 run 目录下的 CSV，
#    跨 run 汇总由使用者自行完成
```

**路径约定**：数据集根目录与 `runs/` 根目录有原项目布局的默认值，可分别用环境变量 `CRACK_DATA_ROOT`（数据集根）、`CRACK_SEG_RUNS`（seg 训练产物根）、`CRACK_DET_RUNS`（seg→det 伪 run 输出根）覆盖，无需修改源码。

评估口径要点（论文 3.2 节）：所有 box 指标一律走 `DetectionValidator`，seg 权重必须先做键转移，且 box 头键匹配率必须为 100%，否则脚本直接报错退出；另在任一 run 上用“直接对 seg 模型取 `.box`”复核，两条路径差值需 ≤ 0.1 点。

---

## 7. 已剔除的内容

为只保留与结论直接相关的代码，以下内容**未纳入**本目录：

- **结构选型阶段的候选实现**：若干候选注意力模块与卷积变体，仅用于结构比较，最终配方均未采用。
- **未进入论文配方的辅助监督与数据扰动分支**：相关类与开关已从交付版本中物理删除，损失向量恒为 3 维，不留“默认关闭”的死代码。
- **早期一次性脚本与产物**：批量评估驱动、探针与绘图脚本、训练日志与检查点目录。

因上述剔除，`overlay/` 只覆盖 7 个文件；其余模块沿用 ultralytics 8.4.81 原版实现。

---

## 8. 自检记录

在 ultralytics 8.4.81 + torch 2.11.0 环境下，将本目录 `overlay/` 覆盖到干净的包副本后构建三条臂：

| 配置 | 参数量 | 640×640 前向 |
|---|---|---|
| `yolov8-ema-crack-nodup.yaml` | 3,011,715 | 通过，输出 `(1, 64, 8400)` |
| `yolov8-ema-crack-nodup-seg.yaml` | 3,264,483 | 通过，输出 `(1, 64, 8400)` |
| `yolov8-ema-only-seg.yaml` | 3,264,483 | 通过，输出 `(1, 64, 8400)` |

两个参数量与论文 4.1、5.1 节完全一致；去 GN 臂与含 GN 臂参数量相同，与论文 2.3 节“参数中性”的表述相符。
