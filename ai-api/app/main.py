from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from .inference import Segmenter
from .weight import WeightPredictor

MAX_UPLOAD_BYTES = 15 * 1024 * 1024

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    seg = Segmenter()          # loaded once per process, never per request
    seg.warmup()
    STATE["seg"] = seg
    # Stage 2 is optional: without a trained ensemble the service still
    # segments and keeps reporting weight_available=false.
    if WeightPredictor.available():
        wp = WeightPredictor()
        wp.warmup()
        STATE["weight"] = wp
    yield
    STATE.clear()


app = FastAPI(title="BovWeight AI Service", version="2.0.0", lifespan=lifespan)


@app.get("/")
@app.get("/health")
def health():
    seg = STATE.get("seg")
    return {
        "status": "ok" if seg else "loading",
        "model_version": seg.model_version if seg else None,
        "classes": seg.names if seg else None,
        "weight_model_version": STATE["weight"].version if "weight" in STATE else None,
    }


async def _read_image(file: UploadFile) -> np.ndarray:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "ไฟล์ที่ส่งมาไม่ใช่รูปภาพ")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "ไฟล์ภาพใหญ่เกิน 15 MB")

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "อ่านไฟล์ภาพไม่สำเร็จ")
    return img


def _weight(img_bgr: np.ndarray) -> dict:
    return STATE["weight"].predict(Image.fromarray(img_bgr[:, :, ::-1]))


@app.post("/api/v1/segment")
async def segment(file: UploadFile = File(...)):
    img = await _read_image(file)
    seg = STATE.get("seg")
    if seg is None:
        raise HTTPException(503, "โมเดลยังโหลดไม่เสร็จ")

    result = seg.analyze(img)

    # Refuse rather than return a confident-looking empty result.
    if result["quality"]["flag"] == "no_animal":
        raise HTTPException(422, "ไม่พบตัวสัตว์ในภาพ กรุณาถ่ายใหม่ให้เห็นเต็มตัว")

    result["weight"] = None
    if "weight" in STATE:
        result["weight"] = _weight(img)
        result["quality"]["weight_available"] = True
        result["quality"]["weight_note"] = (
            "ค่าประมาณจากภาพ (ทดลอง) ยังไม่มีวัตถุอ้างอิงขนาด "
            f"คลาดเคลื่อนเฉลี่ยราว ±{result['weight']['typical_error_kg']:.0f} กก. "
            "ควรชั่งจริงก่อนซื้อขาย"
        )
    return result


@app.post("/api/v1/predict-weight")
async def predict_weight(file: UploadFile = File(...)):
    img = await _read_image(file)
    if "weight" not in STATE:
        raise HTTPException(503, "ยังไม่มีโมเดลประเมินน้ำหนัก")
    return _weight(img)
