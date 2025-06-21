from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image
import base64
from typing import List
import os
import uvicorn

app = FastAPI(title="YOLO Object Detection API", version="1.0.0")

# 정적 파일 (HTML) 서빙
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# CORS 허용 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
try:
    model = YOLO('yolov8n.pt')  # 가장 가벼운 모델 (RAM 초과 방지)
except Exception as e:
    print(f"모델 로드 실패: {e}")
    model = None

# 이미지 처리 함수
def process_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 실패: {str(e)}")

# 감지 결과 이미지 생성
def draw_detections(image: np.ndarray, results) -> str:
    annotated_image = image.copy()
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            class_name = model.names[class_id]
            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    pil_img = Image.fromarray(annotated_image)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

@app.get("/api/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않음")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 허용됨")
    image = process_image(await file.read())
    results = model(image)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            detections.append({
                "class_name": model.names[class_id],
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
            })
    annotated = draw_detections(image, results)
    return {
        "success": True,
        "detections": detections,
        "annotated_image": f"data:image/jpeg;base64,{annotated}"
    }

# FastAPI 실행 (PORT 환경변수로 실행 – Render 전용)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
