"""Stage 1 of the pipeline: YOLO11l-seg -> per-zone masks -> pixel geometry.

Stage 2 (pixels -> kilograms) does not exist yet: it needs a scale reference in
the frame and a regression model trained on weighed animals. This module is
careful to report pixels only.
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from . import geometry, overlay

ROOT = Path(__file__).resolve().parent.parent
# Overridable so a new checkpoint can be rolled out (or rolled back) by changing
# one environment variable instead of editing code.
MODEL_PATH = ROOT / "models" / os.getenv("SEGMENT_MODEL", "segment_v2.pt")
CLASS_MAP_PATH = ROOT / "class_map.json"

IMGSZ = 640          # must match training; see checkpoint train_args
CONF = 0.25
IOU = 0.7


class Segmenter:
    def __init__(self) -> None:
        cfg = json.loads(CLASS_MAP_PATH.read_text(encoding="utf-8"))
        self.zone_meta: dict[str, dict] = cfg["zones"]
        self.model = YOLO(str(MODEL_PATH))
        self.names: dict[int, str] = self.model.names
        # Derived from the checkpoint actually loaded, never from a hand-edited
        # constant - a stale version string in stored measurements would make
        # them impossible to interpret later.
        self.model_version: str = self._version()

        expected = set(self.zone_meta)
        actual = set(self.names.values())
        if expected != actual:
            raise RuntimeError(
                f"class_map.json describes {sorted(expected)} but "
                f"{MODEL_PATH.name} has {sorted(actual)}"
            )

    def _version(self) -> str:
        stamp = ""
        try:
            stamp = str(self.model.ckpt.get("date", ""))[:10]
        except Exception:
            pass
        return f"{MODEL_PATH.stem}@{stamp}" if stamp else MODEL_PATH.stem

    def warmup(self) -> None:
        """First inference allocates buffers and is ~3x slower; pay that cost at
        startup instead of on a user's first scan."""
        self.model.predict(
            np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8), imgsz=IMGSZ, verbose=False
        )

    def analyze(self, image_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        h, w = image_bgr.shape[:2]

        res = self.model.predict(
            image_bgr,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            retina_masks=True,   # masks at original resolution -> areas stay exact
            verbose=False,
        )[0]

        # --- collect one merged mask per zone class -------------------------
        per_zone: dict[int, dict] = {}
        if res.masks is not None:
            data = res.masks.data.cpu().numpy()          # (N, H, W) float 0..1
            cls = res.boxes.cls.cpu().numpy().astype(int)
            conf = res.boxes.conf.cpu().numpy()
            for m, c, cf in zip(data, cls, conf):
                binary = (m > 0.5).astype(np.uint8)
                if binary.shape != (h, w):               # safety net
                    binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
                slot = per_zone.setdefault(c, {"mask": np.zeros((h, w), np.uint8),
                                               "conf": 0.0, "instances": 0})
                slot["mask"] |= binary                   # same zone, several blobs
                slot["conf"] = max(slot["conf"], float(cf))
                slot["instances"] += 1

        # --- whole body ------------------------------------------------------
        # Other animals caught at the edge of the frame are excluded here; every
        # zone is then clipped to the chosen animal so areas stay consistent.
        merged = geometry.union_mask([z["mask"] for z in per_zone.values()], (h, w))
        total_mask, other_animals = geometry.primary_component(merged)
        for slot in per_zone.values():
            slot["mask"] &= total_mask
        total_area = geometry.mask_area(total_mask)

        total = geometry.shape_metrics(total_mask) if total_area else {"area_px": 0}
        total["area_ratio"] = round(total_area / (w * h), 4)
        total["depth_profile"] = geometry.profile_depths(total_mask) if total_area else []

        # --- per zone --------------------------------------------------------
        zones, found_ids, overlay_zones = [], [], []
        for raw_name, meta in self.zone_meta.items():
            cls_id = next((i for i, n in self.names.items() if n == raw_name), None)
            hit = per_zone.get(cls_id)
            area = geometry.mask_area(hit["mask"]) if hit else 0
            # `share` is the payload's most useful number today: a ratio of areas
            # is independent of how far away the camera was.
            share = round(area / total_area, 4) if total_area and area else 0.0

            entry = {
                "id": int(raw_name),
                "key": meta["key"],
                "name_th": meta["name_th"],
                "name_en": meta["name_en"],
                "color": meta["color"],
                "found": bool(area),
                "confidence": round(hit["conf"], 3) if hit else None,
                # blobs remaining after clipping to the primary animal
                "instances": (
                    int(cv2.connectedComponents(hit["mask"])[0] - 1) if area else 0
                ),
                "area_px": area,
                "share": share,
            }
            if area:
                entry.update({
                    "bbox": geometry.bbox(hit["mask"]),
                    "centroid": geometry.centroid(hit["mask"]),
                })
                found_ids.append(int(raw_name))
                overlay_zones.append({
                    "mask": hit["mask"], "color": meta["color"],
                    "name_en": meta["name_en"], "share": share,
                })
            zones.append(entry)

        missing = [int(k) for k in self.zone_meta if int(k) not in found_ids]

        return {
            "success": True,
            "image": {"width": w, "height": h},
            "quality": self._quality(missing, total, other_animals),
            "total": total,
            "zones": zones,
            "overlay_png_base64": overlay.render(image_bgr, overlay_zones) if overlay_zones else None,
            "model": {
                "version": self.model_version,
                "imgsz": IMGSZ,
                "conf": CONF,
                "arch": "yolo11l-seg",
            },
            "inference_ms": int((time.perf_counter() - t0) * 1000),
        }

    @staticmethod
    def _quality(missing: list[int], total: dict, other_animals: int = 0) -> dict:
        """A clean side-on shot shows all seven zones. Missing zones are a
        cheap, built-in signal that the photo is unusable - no extra heuristics
        needed."""
        found = 7 - len(missing)
        reasons: list[str] = []   # things that make the measurement unreliable
        notes: list[str] = []     # things worth telling the user, but not faults
        if found == 0:
            flag = "no_animal"
            reasons.append("ไม่พบตัวสัตว์ในภาพ")
        else:
            if missing:
                reasons.append(f"ตรวจไม่พบ {len(missing)} โซน อาจถ่ายเฉียงหรือถูกบัง")
            aspect = total.get("aspect")
            if aspect is not None and not 1.2 < aspect < 3.2:
                reasons.append("สัดส่วนลำตัวผิดปกติ สัตว์อาจยืนไม่ขนานกับกล้อง")
            if total.get("area_ratio", 0) < 0.06:
                reasons.append("ตัวสัตว์เล็กเกินไปในเฟรม ควรเข้าใกล้กว่านี้")
            if other_animals:
                # Normal in a market or a herd - not a defect, just worth saying.
                notes.append(
                    f"พบสัตว์อื่นอีก {other_animals} ตัวในภาพ ระบบวัดเฉพาะตัวที่ใหญ่ที่สุด"
                )
            flag = "ok" if not reasons else ("partial" if found >= 5 else "poor")

        return {
            "zones_found": found,
            "zones_missing": missing,
            "other_animals": other_animals,
            "notes": notes,
            "flag": flag,
            "reasons": reasons,
            # Honest statement of capability - the UI shows this verbatim.
            "weight_available": False,
            "weight_note": "ยังไม่รองรับการคำนวณน้ำหนัก ต้องมีวัตถุอ้างอิงขนาดในภาพและโมเดล regression (ขั้นที่ 2)",
        }
