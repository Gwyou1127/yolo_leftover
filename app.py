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
from typing import List, Dict
import uvicorn

app = FastAPI(title="YOLO Object Detection API", version="1.0.0")

app.mount("/", StaticFiles(directory="static", html=True), name="static")

# CORS 미들웨어 추가 (웹에서 접근 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포시에는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLO 모델 로드 (모델 경로를 실제 경로로 변경하세요)
try:
    model = YOLO('yolov8n.pt')  # 또는 'your_custom_model.pt'
    print("YOLO 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"모델 로드 실패: {e}")
    model = None

class DetectionResult:
    def __init__(self, class_name: str, confidence: float, bbox: List[float]):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # [x1, y1, x2, y2]

def process_image(image_bytes: bytes) -> np.ndarray:
    """업로드된 이미지를 numpy 배열로 변환"""
    try:
        # PIL Image로 변환
        image = Image.open(io.BytesIO(image_bytes))
        # RGB로 변환 (YOLO는 RGB 형식 사용)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # numpy 배열로 변환
        image_array = np.array(image)
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 실패: {str(e)}")

def draw_detections(image: np.ndarray, results) -> str:
    """감지 결과를 이미지에 그리고 base64로 인코딩"""
    try:
        # 이미지 복사
        annotated_image = image.copy()
        
        # 감지 결과 그리기
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    # 신뢰도
                    confidence = box.conf[0].cpu().numpy()
                    # 클래스 ID
                    class_id = int(box.cls[0].cpu().numpy())
                    # 클래스 이름
                    class_name = model.names[class_id]
                    
                    # 바운딩 박스 그리기
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    
                    # 라벨 텍스트
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # 텍스트 배경 그리기
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                    
                    # 텍스트 그리기
                    cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 6)
        
        # PIL Image로 변환
        annotated_pil = Image.fromarray(annotated_image)
        
        # base64로 인코딩
        buffered = io.BytesIO()
        annotated_pil.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return img_base64
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 주석 처리 실패: {str(e)}")

@app.get("/")
async def root():
    return {"message": "YOLO Object Detection API", "status": "running"}

@app.get("/api/health")
async def health_check():
    """API 상태 확인"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_info": str(model.model.yaml) if model else None
    }

@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    """이미지에서 객체 감지"""
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO 모델이 로드되지 않았습니다.")
    
    # 파일 형식 확인
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    try:
        # 이미지 읽기
        image_bytes = await file.read()
        image = process_image(image_bytes)
        
        # YOLO 모델로 객체 감지
        results = model(image)
        
        # 결과 파싱
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    detections.append({
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })
        
        # 주석이 달린 이미지 생성
        annotated_image_b64 = draw_detections(image, results)
        
        return JSONResponse({
            "success": True,
            "detections": detections,
            "detection_count": len(detections),
            "annotated_image": f"data:image/jpeg;base64,{annotated_image_b64}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"객체 감지 실패: {str(e)}")

@app.post("/detect-batch")
async def detect_objects_batch(files: List[UploadFile] = File(...)):
    """여러 이미지에서 일괄 객체 감지"""
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO 모델이 로드되지 않았습니다.")
    
    results = []
    
    for file in files:
        if not file.content_type.startswith("image/"):
            continue
        
        try:
            image_bytes = await file.read()
            image = process_image(image_bytes)
            
            # YOLO 모델로 객체 감지
            detection_results = model(image)
            
            # 결과 파싱
            detections = []
            for result in detection_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = model.names[class_id]
                        
                        detections.append({
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)]
                        })
            
            results.append({
                "filename": file.filename,
                "detections": detections,
                "detection_count": len(detections)
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse({
        "success": True,
        "results": results,
        "processed_count": len(results)
    })

@app.get("/classes")
async def get_classes():
    """사용 가능한 클래스 목록 반환"""
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO 모델이 로드되지 않았습니다.")
    
    return {
        "classes": model.names,
        "class_count": len(model.names)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render는 기본적으로 PORT 환경변수를 설정함
    print(f"YOLO Object Detection API 서버를 시작합니다 (포트: {port})...")
    uvicorn.run(app, host="0.0.0.0", port=port)