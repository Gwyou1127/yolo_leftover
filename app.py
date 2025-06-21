import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import random
import io

# 모델 불러오기
model = YOLO('best.pt')  # 모델 경로 수정

# 라벨 색상 고정 안 함 (Gradio처럼 매번 랜덤)
label_colors = []
for _ in range(len(model.names)):
    label_colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

# 이미지 전처리 및 추론
def detect_objects(image_bytes, conf_thres, iou_thres):
    # Gradio와 동일하게 이미지 BGR로 디코딩
    image_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # 추론
    results = model.predict(source=img, conf=conf_thres, iou=iou_thres)[0]
    
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()

    # 이미지 복사본 생성
    annotated_image = img.copy()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        class_id = int(classes[i])
        confidence = float(confidences[i])
        label = model.names[class_id]
        box_color = label_colors[class_id]
        
        # 바운딩박스
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color, 20, cv2.LINE_AA)
        
        # 텍스트 스타일
        label_text = f"{label} {confidence:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_image, (x1, y1-label_height-3), (x1+label_width, y1), box_color, -1)
        cv2.putText(annotated_image, label_text, (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 16)

    # BGR → RGB 변환 후 반환
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return annotated_image

# Streamlit UI
st.set_page_config(page_title="잔반 탐지기 (YOLOv8)", layout="centered")
st.title("🍴 잔반 탐지기 (YOLOv8 Gradio 스타일)")

# 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

# 슬라이더
conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.75, 0.01)
iou = st.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.01)

# 추론
if uploaded_file is not None:
    with st.spinner("객체 감지 중..."):
        image_bytes = uploaded_file.read()
        result_image = detect_objects(image_bytes, conf, iou)
        st.image(result_image, caption="감지 결과", use_column_width=True)
