from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from .inference import Segmenter

MAX_UPLOAD_BYTES = 15 * 1024 * 1024

STATE: dict[str, Segmenter] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    seg = Segmenter()          # loaded once per process, never per request
    seg.warmup()
    STATE["seg"] = seg
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
    }


@app.post("/api/v1/segment")
async def segment(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "ไฟล์ที่ส่งมาไม่ใช่รูปภาพ")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "ไฟล์ภาพใหญ่เกิน 15 MB")

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "อ่านไฟล์ภาพไม่สำเร็จ")

    seg = STATE.get("seg")
    if seg is None:
        raise HTTPException(503, "โมเดลยังโหลดไม่เสร็จ")

    result = seg.analyze(img)

    # Refuse rather than return a confident-looking empty result.
    if result["quality"]["flag"] == "no_animal":
        raise HTTPException(422, "ไม่พบตัวสัตว์ในภาพ กรุณาถ่ายใหม่ให้เห็นเต็มตัว")

    return result
