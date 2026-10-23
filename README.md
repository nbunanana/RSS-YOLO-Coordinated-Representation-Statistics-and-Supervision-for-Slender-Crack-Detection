# RSS-YOLO Core Code

This directory collects the code that produces every numerical result reported in the paper RSS-YOLO: Coordinated Representation, Statistics and Supervision for Slender Crack Detection. The material follows four stages: model structure, loss, training and evaluation.

It contains no data. There is no dataset, no image or annotation, no pretrained weight, and no training log or result directory. Only the implementation logic and the configuration files are included.

---

## 1. Relation to the paper

The paper uses three configurations, each of which adds to the `yolov8n` baseline. The corresponding files reside in `configs/`:

| Term used in the paper | Configuration file | Structural difference | Parameters (measured) |
|---|---|---|---|
| Deployment form, the detection structure that underlies Tables 2 and 3 and every mAP value | `yolov8-ema-crack-nodup.yaml` | P3 to `C2f_EMA`, P4/P5 neck to `C2f_GN`, `Detect` head | 3,011,715 |
| Training form, which carries the training-time mask branch | `yolov8-ema-crack-nodup-seg.yaml` | As above, with the last layer replaced by `Segment` and proto taken from P3 | 3,264,483 |
| GN-free control, the `+EMA+Mask` class of control of Section 4.1 | `yolov8-ema-only-seg.yaml` | `C2f_GN` at layers 12, 18 and 21 reverted to the standard `C2f` | 3,264,483 |

Both parameter counts agree digit for digit with the values reported in Sections 4.1 and 5.1 of the paper, which makes them the quickest check that the environment is correct. `verify.py` performs that check.

`C2f_GN` and `C2f` carry an identical number of parameters, because GN and BN both apply a per-channel affine transform and are therefore parameter neutral. Capacity is consequently isolated between the GN-free configuration and the full recipe, so a difference between the two configurations can be attributed to the mechanism rather than to parameter count. This point is the basis of the argument in Section 2.3 of the paper.

---

## 2. Directory layout

```
code/
├── overlay/ultralytics/          # modified files that overwrite the same paths in the ultralytics package
│   ├── nn/modules/attention.py   # EMA multi-scale attention
│   ├── nn/modules/block.py       # C2f_EMA and C2f_GN
│   ├── nn/modules/conv.py        # contains Conv_GN and Bottleneck_GN
│   ├── nn/modules/head.py        # Segment head with proto taken from P3 (CRACK_PROTO_LVL)
│   ├── nn/modules/__init__.py    # exports
│   ├── nn/tasks.py               # parse_model registration
│   └── utils/loss.py             # CRACK_SEG_W and CRACK_SEG_SHUFFLE
├── configs/                      # model configuration files
├── scripts/
│   ├── resplit_dataset.py        # re-splits Crack-Seg at 7:1:2 (split seed set by --seed)
│   └── seg2det_eval.py           # seg-to-det weight-key transfer, box-head 100% match assertion, dual-path check
├── verify.py                     # self-test: builds the three configurations and checks parameter counts
└── requirements.txt
```

---

## 3. Key implementations

### 3.1 Representation level: `C2f_EMA` (`nn/modules/block.py`)

`C2f_EMA` inherits from `C2f` and adds a single EMA cross-spatial recalibration after the `cv2` fusion:

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

The channel flow and the number of bottlenecks stay unchanged, and the insertion point is decided by the configuration. The paper places the module at layer 15 only, that is, at P3. Inside EMA the channels are grouped and the weights are shared within a group, while two branches pool entire rows and columns with 1×k and k×1 kernels. The factor is 8 and the module holds 672 parameters.

### 3.2 Statistics level: `C2f_GN` (`nn/modules/block.py`)

`C2f_GN` is layer-by-layer isomorphic to `C2f`. Only `Conv` is replaced by `Conv_GN` and `Bottleneck` by `Bottleneck_GN`, which moves normalisation from the batch dimension to statistics computed within channel groups. The parameter count is exactly the same as in the BN version.

### 3.3 Supervision level: auxiliary mask supervision (`utils/loss.py`)

Two lines implement the switch that lets the mask supervision term be switched off independently, which is the property described in Section 2.4 of the paper:

```python
CRACK_SEG_W = float(os.environ.get("CRACK_SEG_W", "1.0"))
...
loss[4] *= self.hyp.box * CRACK_SEG_W   # semantic term, inactive because this dataset carries no such label
loss[1] *= self.hyp.box * CRACK_SEG_W   # mask term
```

In the upstream implementation the mask term and box regression share the same `hyp.box`, so closing the mask through `hyp.box=0` also zeroes box regression, and the result is not a control that detaches the mask alone. A coefficient that acts on the mask term only is therefore introduced separately:

- `CRACK_SEG_W=1.0`, the default, is bit-identical to the original and serves the primary configuration;
- `CRACK_SEG_W=0` detaches the mask supervision alone, leaving box, classification and DFL supervision unchanged.

The shuffled control of Section 4.2 of the paper is triggered by `CRACK_SEG_SHUFFLE=1`. Within one batch it permutes the correspondence between the predicted masks and the ground-truth instances along the sample dimension without fixed points, and every other setting is identical to the primary configuration.

---

## 4. Environment and installation

Dependencies are listed in `requirements.txt`. The ultralytics version is **8.4.81**, and the files in `overlay/` are copied over the package along the same relative paths:

```bash
pip install -r requirements.txt
# copy the modified files over the installed package (example: a venv on Linux or macOS)
cp -r overlay/ultralytics/* "$(python -c 'import ultralytics,os;print(os.path.dirname(ultralytics.__file__))')"/
python verify.py
```

`verify.py` builds the three configurations and asserts parameter counts of 3,011,715, 3,264,483 and 3,264,483. Agreement with the values listed in this directory confirms that the overlay has been applied correctly.

---

## 5. Key environment variables

The `CRACK_*` names are environment variables. They are **not written into `args.yaml`** and must be injected explicitly in the same process that starts training or evaluation, otherwise results are not comparable:

| Variable | Value used in the paper | Effect |
|---|---|---|
| `CRACK_IOU` | `ciou` | Type of box regression loss. The code defaults to `siou`, whereas **the paper protocol specifies `ciou`, which must be overridden explicitly** |
| `CRACK_NORM` | `bn` | Normalisation of the backbone; the paper keeps BN |
| `CRACK_PROTO_LVL` | `0` | proto taken from P3, that is, `x[0]` |
| `CRACK_SEG_W` | `1.0` / `0` | Coefficient of the mask term; 1 serves the primary configuration and 0 gives the mask-free control |
| `CRACK_SEG_SHUFFLE` | `0` / `1` | 1 gives the shuffled control of Section 4.2 of the paper |
| `CRACK_SEG_SHUFFLE_SEED` | `0` | Permutation seed of the shuffled control, held fixed so that successive runs reproduce the same permutation |

The training hyperparameters (`epochs=300`, `imgsz=640`, `batch=16`, `mosaic=0.0`, `deterministic=true`, `pretrained=yolov8n.pt`, `freeze=null`) appear in the training command of Section 6.

---

## 6. Reproduction steps

```bash
# 1) re-split the data at 7:1:2 (the split seed must stay fixed to reproduce the same split) and write datasets/crack-seg-712 with split_manifest.csv
python scripts/resplit_dataset.py --seed <SEED>

# 2) train (example: the primary configuration)
CRACK_IOU=ciou CRACK_NORM=bn CRACK_PROTO_LVL=0 CRACK_SEG_W=1.0 \
python -c "from ultralytics import YOLO; YOLO('configs/yolov8-ema-crack-nodup-seg.yaml').train(data='crack-seg-712.yaml', epochs=300, imgsz=640, batch=16, seed=<SEED>, mosaic=0.0, deterministic=True, pretrained='yolov8n.pt')"

# 3) evaluate under the same protocol: transfer the seg weight keys to the det structure, assert a 100% match on the box head, then report val and test mAP
python scripts/seg2det_eval.py --runs <RUN_DIR> --det-yaml configs/yolov8-ema-crack-nodup.yaml --splits val test --dual-path

# 4) run step 3 on any additional run directory; the user carries out the aggregation of the per-run CSV files
```

**Path conventions.** The dataset root and the `runs/` root carry defaults that follow the original project layout. The environment variables `CRACK_DATA_ROOT` for the dataset root, `CRACK_SEG_RUNS` for the seg training output root and `CRACK_DET_RUNS` for the output root of the seg-to-det pseudo runs override those defaults without any change to the source.

The training seed is supplied through the `seed=` argument shown above, and the package presets no value for it.

Two points of the evaluation protocol follow Section 3.2 of the paper. Every box metric passes through `DetectionValidator`, the seg weights must first undergo key transfer, and the match rate on the box-head keys must reach 100%, otherwise the script reports an error and exits. In addition, one run is rechecked by reading `.box` directly from the seg model, and the difference between the two paths must stay within 0.1 point.

---

## 7. Content left out

To keep only the code that bears directly on the conclusions, the following material is **not included** in this directory:

- Design variants left outside the final recipe, namely several alternative attention modules and convolution variants.
- Auxiliary supervision and data-perturbation branches that did not enter the paper recipe. The corresponding classes and switches are physically removed from the delivered version rather than left switched off, and the loss vector is always three-dimensional.
- Early one-off scripts and artefacts, covering batch evaluation drivers, probe and plotting scripts, training logs and checkpoint directories.

With that material removed, `overlay/` covers 7 files only, and every other module keeps the stock implementation of ultralytics 8.4.81.

---

## 8. Self-test record

Under ultralytics 8.4.81 and torch 2.11.0, `overlay/` was applied to a clean copy of the package and the three configurations were built:

| Configuration | Parameters | 640×640 forward pass |
|---|---|---|
| `yolov8-ema-crack-nodup.yaml` | 3,011,715 | passes, output `(1, 64, 8400)` |
| `yolov8-ema-crack-nodup-seg.yaml` | 3,264,483 | passes, output `(1, 64, 8400)` |
| `yolov8-ema-only-seg.yaml` | 3,264,483 | passes, output `(1, 64, 8400)` |

Both parameter counts agree exactly with Sections 4.1 and 5.1 of the paper. The GN-free configuration and the GN configuration carry the same number of parameters, in agreement with the parameter-neutral statement of Section 2.3.
