# ml/ — weight regression (stage 2)

Replaces the approach in `Predition_Weight.ipynb`, whose EfficientNet-V2-M
reached **MAE 79.91 kg / RMSE 92.66 kg** on the test split — worse than
answering with the mean weight every time (R² below zero).

## Why that happened

| # | Cause | Evidence |
|---|-------|----------|
| 1 | `Resize((384,384))` on a bbox crop erases absolute size **and** the length-to-height ratio | predictions collapse to 285–425 kg while the truth spans 300–469 kg |
| 2 | `Height`, `L`, `Age`, `Category` in `cow.csv` are never used | no reference to those columns anywhere in the notebook |
| 3 | 54M parameters for ~220 distinct images | train RMSE 28 vs valid RMSE 91 at the best epoch |
| 4 | `OneCycleLR(epochs=400)` + early stop at 187 | LR was still 0.00043, near peak, never annealed |
| 5 | Target not standardised, head bias starts at 0 | epoch 1 RMSE = 406 ≈ the mean weight |
| 6 | 28-image valid / 28-image test, split by file not by animal | one sample moves RMSE by several kg; augmented copies can leak |

## Pipeline

```
00_extract_geometry.py   original images → bbox / 7-zone areas in real pixels
01_baseline_tabular.py   Height + L + Age → weight   (run this FIRST)
02_train_hybrid.py       image ⊕ geometry ⊕ measurements, GroupKFold by animal

Multitask_Weight_Experiment.ipynb   Colab: multi-task + crop/no-crop matrix
build_notebook.py                   regenerates that notebook from plain Python
```

### The Colab notebook

`Multitask_Weight_Experiment.ipynb` answers two questions in one run:

**Does multi-task help?** One encoder predicts `weight`, `height`, `length`,
`age`, `breed` and the scale-free `L/H` ratio. Task losses are balanced by
learned uncertainty, so a target the model cannot fit down-weights itself.
Expect `ratio_lh` and `breed` to learn well and `Height`/`L` in centimetres not
to — that split *is* the scale problem, visible directly in the auxiliary heads.

**Does cropping help or hurt?** Six variants under identical folds and seeds:

| | no geometry | + geometry (bbox in original px) |
|---|---|---|
| full frame | A | D |
| tight crop | B | E |
| crop + 20% context | C | F |

plus `E- single-task` as the ablation. **A beating B is the decisive result**:
it would mean the binding constraint is scale, not background clutter.

### Run

```bash
# 0. geometry from the uncropped images (reuses ai-api's own segmenter)
python ml/00_extract_geometry.py \
    --images /content/cow_dataset \
    --csv    /content/cow_dataset/cow_weight_dataset.csv \
    --model  ai-api/models/segment.pt \
    --out    ml/geometry.csv

# 1. the bar every image model has to clear — seconds, no GPU
python ml/01_baseline_tabular.py --csv /content/cow.csv

# 2. hybrid model
python ml/02_train_hybrid.py \
    --data     /content/cow_dataset \
    --geometry ml/geometry.csv \
    --cow-csv  /content/cow.csv \
    --folds 5 --epochs 80
```

### Deployed model: feature ensemble (`03_train_weight_ensemble.py`)

What ai-api serves (`ai-api/app/weight.py`), from `Predition_Weight_V3 (1).ipynb` cells 35-39:
the four fine-tuned backbones in `ai-api/models/predict_weight/*.pth` are frozen,
their flip-averaged features are concatenated with YOLO11n box geometry
(3208 features), and SVR, Ridge and a multi-task MLP are fitted on them.
The served weight is the mean of the three.

```bash
# from the repo root, in the ai-api image (same code path as serving)
docker compose build ai-api
docker run --rm -v "$PWD/ai-api:/work/ai-api" -v "$PWD/ml:/work/ml" \
    -v /path/to/dataset:/data -w /work bovweightai-ai-api sh -c \
    "pip install -q pandas && python ml/03_train_weight_ensemble.py \
       --images /data/cow_dataset --cow-csv /data/cow.csv --cache /data/weight_features.npz"
docker compose up -d ai-api     # picks up models/predict_weight/ensemble/ on start
```

Outputs `ai-api/models/predict_weight/ensemble/` (`svr.joblib`, `ridge.joblib`,
`mlp.pt`, `manifest.json`, `oof_predictions.csv`). Delete that folder and
ai-api falls back to segmentation only.

**Read the metrics in `manifest.json` carefully.** The backbones were fine-tuned
with weight as a target on fold 0 of an 8-fold split, i.e. on ~7/8 of these
animals, so cross-validating the heads on all rows is optimistic. The script
rebuilds that split, checks it against each checkpoint's stored `val_mae`, and
reports `unseen_by_backbones` separately - that is the honest figure, and it is
what the app shows as the typical error. Also note that most `cow.csv` weights
equal `5.854*L - 595.98` exactly (a tape formula, not a scale); the script
reports formula and weighed-looking labels separately.

Start with step 1. If three tape measurements beat the CNN — and they very
likely will — that settles the diagnosis: the problem is the inputs, not the
architecture.

## What changed, and why

- **`LetterboxPad` instead of `Resize((n,n))`** — pads to a square rather than
  squashing, so body proportion survives.
- **Geometry as model input** — bbox and zone areas in original-image pixels
  are the only channel through which absolute size can reach the model.
- **`GroupKFold` on animal id** — `normalise_id()` strips Roboflow's `-1`
  augmentation suffix so every photo of one cow lands in one fold. Uses all
  ~140 animals for evaluation instead of 28 images.
- **Standardised target, head bias at the mean** — no wasted epochs.
- **`SmoothL1Loss`** — one mislabelled cow cannot dominate the gradient.
- **EfficientNet-B0 (5M)** instead of V2-M (54M), dropout 0.4, weight decay 1e-2.
- **`CosineAnnealingLR`** — stopping early still leaves an annealed LR.
- **Horizontal-flip TTA** and **per-animal averaging** at evaluation.
- **Metrics a farmer can read** — MAPE, bias, % within ±10%, plus a
  predict-the-mean row so a negative R² is impossible to overlook.

## The ceiling nobody can train past

Absolute size is not recoverable from a single photo with no object of known
size in frame. Shape ratios survive any crop or resize; **centimetres do not.**

Three ways out, cheapest first:

1. **ArUco marker** on paper in the plane of the animal's body — near-exact,
   effectively free.
2. Predict `Height` and `L` from the image, then apply the fitted
   `W ≈ a·(H²·L) + b` rule from step 1. Two easier problems instead of one hard
   one, and each stage is inspectable.
3. Depth capture (LiDAR/ToF), which limits which phones can be used.

Until one of these is in the capture protocol, expect a hard floor on accuracy
no amount of training compute will move.
