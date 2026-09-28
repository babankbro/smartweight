"""Stage 2: photo -> body weight (kg).

Pipeline (mirrors `ml/Predition_Weight_V3 (1).ipynb`, cells 35-39):

    photo --stretch--> 640x640 RGB          (the Roboflow export was stretched to 640x640)
      |-- YOLO11n (COCO) -> largest box -> 8 geometry features
      |-- 4 fine-tuned backbones, flip-averaged -> 768 + 1280 + 384 + 768 features
      '-- concat (3208) -> SVR | Ridge | multi-task MLP (| backbones' own
          weight head, "direct") -> manifest-weighted sum = weight

Training (`ml/03_train_weight_ensemble.py`) imports this module, so the
preprocessing used to fit the heads and the one used to serve them are the same
code - a mismatch here would silently shift every prediction.
"""

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
# A model set is a folder of backbone .pth files plus an ensemble sub-folder
# (manifest + heads). Switch sets with env vars, no rebuild:
#   cattle only (2026-09-27): WEIGHT_DIR=predict_weight     WEIGHT_ENSEMBLE=ensemble
#   cow+buffalo (2026-09-28): WEIGHT_DIR=predict_weight_v2  WEIGHT_ENSEMBLE=ensemble_cowbuff
WEIGHT_DIR = ROOT / "models" / os.getenv("WEIGHT_DIR", "predict_weight")
ENSEMBLE_DIR = WEIGHT_DIR / os.getenv("WEIGHT_ENSEMBLE", "ensemble")
# Stock COCO yolo11n - shared by every set, so it is looked up in the set
# first and then in the original folder.
DETECTOR_PATH = next(
    (p for p in (WEIGHT_DIR / "yolo11n.pt", ROOT / "models" / "predict_weight" / "yolo11n.pt") if p.exists()),
    WEIGHT_DIR / "yolo11n.pt",
)

INPUT_SIZE = 640     # every training image is a 640x640 stretch; serve the same
FEAT_IMG = 336       # IMG_SIZE the backbones were fine-tuned at (notebook cell 25)
DET_CONF = 0.25

# Concatenation order matters: the heads were fitted on exactly this layout.
BACKBONE_KEYS = ["convnext_t", "effv2_s", "dinov2_s", "clip_b16"]
GEOM_COLS = ["bbox_w", "bbox_h", "bbox_area_ratio", "bbox_aspect",
             "bbox_w_rel", "bbox_h_rel", "img_w", "img_h"]
REG_TARGETS = ["weight", "Height_y", "L_y", "age_years", "ratio_lh"]
N_BREEDS = 4         # only needed to rebuild the checkpoint's classifier head

# The multi-task MLP predicts every REG_TARGET, not just weight. API field name
# and unit for the non-weight ones:
AUX_TARGETS = {
    "Height_y": ("height_cm", "cm"),
    "L_y": ("length_cm", "cm"),
    "age_years": ("age_years", "years"),
    "ratio_lh": ("ratio_lh", ""),
}

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class LetterboxPad:
    """Resize the long edge and pad to a square (fill 114), as in training."""

    def __init__(self, size: int, fill: int = 114):
        self.size, self.fill = size, fill

    def __call__(self, im: Image.Image) -> Image.Image:
        w, h = im.size
        s = self.size / max(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        c = Image.new("RGB", (self.size, self.size), (self.fill,) * 3)
        c.paste(im, ((self.size - im.size[0]) // 2, (self.size - im.size[1]) // 2))
        return c


FEAT_TF = transforms.Compose([LetterboxPad(FEAT_IMG), transforms.ToTensor(), NORM])


def to_model_input(im: Image.Image) -> Image.Image:
    return im.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)


class WeightNet(nn.Module):
    """Architecture of the fine-tuned checkpoints (notebook cell 26). Only the
    backbone is used for features, but the full net is rebuilt so the
    checkpoint loads strictly and a wrong file fails loudly."""

    def __init__(self, backbone: str, n_feats=len(GEOM_COLS), n_breeds=N_BREEDS,
                 n_reg=len(REG_TARGETS), dropout=0.4):
        super().__init__()
        dyn = "dinov2" in backbone or "clip" in backbone
        self.backbone = timm.create_model(
            backbone, pretrained=False, num_classes=0,
            **({"dynamic_img_size": True} if dyn else {}),
        )
        d_img = self.backbone.num_features
        self.tab = nn.Sequential(
            nn.BatchNorm1d(n_feats), nn.Linear(n_feats, 64), nn.SiLU(),
            nn.Dropout(dropout / 2), nn.Linear(64, 64), nn.SiLU(),
        )
        self.trunk = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_img + 64, 256), nn.SiLU())
        self.reg = nn.Linear(256, n_reg)
        self.breed = nn.Linear(256, n_breeds)
        self.log_var = nn.Parameter(torch.zeros(n_reg + 1))

    def forward(self, img, feats):
        z = torch.cat([self.backbone(img), self.tab(feats)], 1)
        z = self.trunk(z)
        return self.reg(z), self.breed(z)


class FrozenMultitaskMLP(nn.Module):
    """Head on the concatenated features (notebook cell 33)."""

    def __init__(self, in_features: int, n_reg=len(REG_TARGETS), dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(dropout / 2),
            nn.Linear(256, n_reg),
        )
        self.log_var = nn.Parameter(torch.zeros(n_reg))

    def forward(self, x):
        return self.net(x)


def load_finetuned(key_or_file: str, device="cpu") -> tuple[WeightNet, dict]:
    name = key_or_file if key_or_file.endswith(".pth") else f"{key_or_file}_multitask_fold1.pth"
    ckpt = torch.load(WEIGHT_DIR / name, map_location=device, weights_only=True)
    # cow+buffalo checkpoints carry their own breed count
    net = WeightNet(ckpt["backbone"], n_breeds=ckpt.get("n_breeds", N_BREEDS))
    net.load_state_dict(ckpt["state_dict"])
    return net.eval().to(device), ckpt


class FeatureExtractor:
    """Photo -> (geometry dict, 3208-d feature vector before geometry scaling)."""

    def __init__(self, device="cpu", files: list[str] | None = None):
        from ultralytics import YOLO

        self.device = device
        self.detector = YOLO(str(DETECTOR_PATH))
        loaded = [load_finetuned(f, device) for f in (files or BACKBONE_KEYS)]
        self.nets = dict(zip(BACKBONE_KEYS, (net for net, _ in loaded)))
        self.wnorm = [ck["norm"]["weight"] for _, ck in loaded]   # (mu, sd) of each net's weight head

    def geometry(self, im640: Image.Image) -> dict:
        """Largest box of ANY class, as the notebook did (it never filtered to
        'cow'); falls back to the full frame when nothing is detected."""
        bgr = np.asarray(im640)[:, :, ::-1]
        res = self.detector.predict(bgr, conf=DET_CONF, verbose=False)[0]
        W, H = im640.size
        x1 = y1 = 0.0
        x2, y2, found = float(W), float(H), False
        if res.boxes is not None and len(res.boxes):
            xy = res.boxes.xyxy.cpu().numpy()
            a = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
            x1, y1, x2, y2 = (float(v) for v in xy[int(a.argmax())])
            found = True
        bw, bh = x2 - x1, y2 - y1
        return dict(found=found, x1=x1, y1=y1, x2=x2, y2=y2,
                    bbox_w=bw, bbox_h=bh, bbox_area_ratio=bw * bh / (W * H),
                    bbox_aspect=bw / bh if bh else 0.0,
                    bbox_w_rel=bw / W, bbox_h_rel=bh / H, img_w=float(W), img_h=float(H))

    @torch.no_grad()
    def backbone_features(self, ims640: list[Image.Image], geom: np.ndarray | None = None):
        """Flip-averaged features. Given scaled `geom`, also returns the nets' own
        weight prediction (kg, mean over nets) - the ensemble's `direct` head."""
        x = torch.stack([FEAT_TF(im) for im in ims640]).to(self.device)
        xf = torch.flip(x, dims=[3])
        parts, direct = [], []
        for net, (mu, sd) in zip(self.nets.values(), self.wnorm):
            fa, fb = net.backbone(x), net.backbone(xf)
            parts.append(((fa + fb) / 2).float().cpu().numpy())
            if geom is not None:
                t = net.tab(torch.tensor(geom, dtype=torch.float32, device=self.device))
                p = sum(net.reg(net.trunk(torch.cat([f, t], 1)))[:, 0] for f in (fa, fb)) / 2
                direct.append(p.float().cpu().numpy() * sd + mu)
        feats = np.concatenate(parts, axis=1)
        return feats if geom is None else (feats, np.mean(direct, axis=0))


def scale_geometry(geom_rows: np.ndarray, stats: dict) -> np.ndarray:
    mu = np.array([stats[c][0] for c in GEOM_COLS])
    sd = np.array([stats[c][1] for c in GEOM_COLS])
    return (geom_rows - mu) / (sd + 1e-8)


class WeightPredictor:
    """Loads the trained ensemble. Construct only if `available()`."""

    @staticmethod
    def available() -> bool:
        return (ENSEMBLE_DIR / "manifest.json").exists()

    def __init__(self, device="cpu"):
        self.man = json.loads((ENSEMBLE_DIR / "manifest.json").read_text(encoding="utf-8"))
        if self.man["backbones"] != BACKBONE_KEYS or self.man["geom_cols"] != GEOM_COLS:
            raise RuntimeError("weight ensemble manifest does not match app/weight.py layout")
        self.fx = FeatureExtractor(device, self.man.get("backbone_files"))
        self.heads = self.man.get("heads", ["svr", "ridge", "mlp"])
        self.weights = self.man.get("weights") or [1 / len(self.heads)] * len(self.heads)
        self.svr = joblib.load(ENSEMBLE_DIR / "svr.joblib")
        self.ridge = joblib.load(ENSEMBLE_DIR / "ridge.joblib")
        ck = torch.load(ENSEMBLE_DIR / "mlp.pt", map_location=device, weights_only=True)
        self.mlp = FrozenMultitaskMLP(ck["in_features"]).eval()
        self.mlp.load_state_dict(ck["state_dict"])
        self.mlp_targets = ck["targets"]
        self.mlp_mu, self.mlp_sd = np.array(ck["y_mu"]), np.array(ck["y_sd"])
        self.version = self.man["version"]
        # Older (cattle-only) manifests have no species list.
        self.species = self.man.get("species", ["cow"])

    def warmup(self) -> None:
        self.predict(Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114)))

    @torch.no_grad()
    def predict(self, im: Image.Image) -> dict:
        t0 = time.perf_counter()
        im640 = to_model_input(im)
        g = self.fx.geometry(im640)
        geom = scale_geometry(np.array([[g[c] for c in GEOM_COLS]]), self.man["geom_stats"])
        direct = None
        if "direct" in self.heads:
            feats, direct = self.fx.backbone_features([im640], geom)
        else:
            feats = self.fx.backbone_features([im640])
        X = np.hstack([feats, geom])

        p_svr = float(self.svr.predict(X)[0])
        p_ridge = float(self.ridge.predict(X)[0])
        mlp_out = self.mlp(torch.tensor(X, dtype=torch.float32))[0].numpy() \
            * self.mlp_sd + self.mlp_mu
        p_mlp = float(mlp_out[0])
        per_model = {"svr": p_svr, "ridge": p_ridge, "mlp": p_mlp}
        if direct is not None:
            per_model["direct"] = float(direct[0])
        preds = [per_model[h] for h in self.heads]

        # Other multi-task outputs. Only the MLP predicts these (SVR / Ridge
        # were fitted on weight alone), so each is a single-model estimate.
        aux_err = self.man.get("aux_typical_error", {})
        measurements = {}
        for i, t in enumerate(self.mlp_targets):
            if t not in AUX_TARGETS:
                continue
            key, unit = AUX_TARGETS[t]
            e = aux_err.get(t, {})
            measurements[key] = {
                "value": round(float(mlp_out[i]), 2 if t == "ratio_lh" else 1),
                "unit": unit,
                "typical_error": e.get("MAE"),
                "r2": e.get("R2"),
            }

        return {
            "kg": round(float(np.dot(self.weights, preds)), 1),
            # how much the heads disagree - not a confidence interval
            "spread_kg": round(float(np.std(preds)), 1),
            "per_model": {k: round(v, 1) for k, v in per_model.items()},
            # share of each head in `kg` (NNLS may zero some out)
            "head_weights": {h: round(float(w), 3) for h, w in zip(self.heads, self.weights)},
            "typical_error_kg": self.man.get("typical_error_kg"),
            "typical_error_by_species": self.man.get("typical_error_by_species"),
            "species": self.species,
            "measurements": measurements,
            "detector_found": g["found"],
            "model_version": self.version,
            "inference_ms": int((time.perf_counter() - t0) * 1000),
        }
