FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first, so the ultralytics install below sees it as satisfied
# and does not drag in the CUDA build.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4

COPY class_map.json .
COPY app ./app
# models/ is mounted as a volume at runtime - a 56 MB weight file has no
# business being baked into the image on every rebuild.

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
