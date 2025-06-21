# YOLOv8 Streamlit 앱 (UX 개선 + 감지 정보 + 영상 지원)

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import os
import torch
import tempfile
import time
import random
import warnings

# 설정
st.set_page_config(page_title="🍴 잔반 탐지기", page_icon="🍴", layout="wide")
os.environ['YOLO_VERBOSE'] = 'False'
warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 모델 로딩
@st.cache_resource(show_spinner=True)
def load_model():
    return YOLO("best.pt")

# 라벨 색 고정
def get_label_colors(model):
    random.seed(42)
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in model.names]

# 감지 통계 출력
def display_stats(detections):
    class_counts = {}
    for det in detections:
        class_counts[det[0]] = class_counts.get(det[0], 0) + 1

    total = len(detections)
    avg_conf = sum([conf for _, conf in detections]) / total if total > 0 else 0
    max_conf = max([conf for _, conf in detections], default=0)

    col1, col2, col3 = st.columns(3)
    col1.metric("총 객체 수", total)
    col2.metric("평균 정확도", f"{avg_conf:.2%}")
    col3.metric("최고 정확도", f"{max_conf:.2%}")

# 이미지 감지
def detect_image(image_bytes, confidence):
    model = load_model()
    label_colors = get_label_colors(model)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)
    results = model.predict(img_array, conf=confidence, iou=0.5)[0].cpu().numpy()

    detections = []
    for i, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        cls = int(results.boxes.cls[i])
        conf = float(results.boxes.conf[i])
        label = model.names[cls]
        color = label_colors[cls]
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_array, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        detections.append((label, conf, [x1, y1, x2, y2]))
    return Image.fromarray(img_array), detections

# 영상 감지
def detect_video(video_file, confidence):
    model = load_model()
    label_colors = get_label_colors(model)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    out_path = tempfile.mktemp(suffix=".mp4")
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=confidence, iou=0.5)[0]
        if results.boxes is not None:
            for i, box in enumerate(results.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                cls = int(results.boxes.cls[i])
                conf = float(results.boxes.conf[i])
                label = model.names[cls]
                color = label_colors[cls]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        out.write(frame)
        frame_idx += 1
        progress.progress(min(frame_idx / total, 1.0))

    cap.release()
    out.release()
    return out_path

# UI 시작
st.title("🍴 잔반 탐지기")
tabs = st.tabs(["📷 이미지 추론", "🎞️ 영상 추론"])

# 이미지 탭
with tabs[0]:
    uploaded_image = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    conf = st.slider("정확도 임계값", 0.1, 1.0, 0.5, step=0.05)
    if uploaded_image:
        image_bytes = uploaded_image.read()
        st.subheader("📸 업로드된 이미지")
        st.image(image_bytes, use_column_width=True)
        with st.spinner("감지 중..."):
            result_img, detections = detect_image(image_bytes, conf)
        st.subheader("🎯 감지 결과")
        st.image(result_img, use_column_width=True)
        st.subheader("📊 감지 통계")
        display_stats(detections)
        st.subheader("📋 객체 상세 정보")
        for i, (label, conf, box) in enumerate(sorted(detections, key=lambda x: -x[1])):
            with st.expander(f"{i+1}. {label} ({conf:.2%})"):
                st.write(f"정확도: {conf:.3f}")
                st.write(f"좌표: {box}")

# 영상 탭
with tabs[1]:
    uploaded_video = st.file_uploader("영상을 업로드하세요", type=["mp4", "mov", "avi"])
    conf_v = st.slider("정확도 임계값", 0.1, 1.0, 0.5, step=0.05, key="video_conf")
    if uploaded_video:
        st.subheader("🎞️ 업로드된 영상")
        st.video(uploaded_video, format="video/mp4")
        with st.spinner("영상 처리 중... 시간이 다소 걸릴 수 있습니다"):
            out_path = detect_video(uploaded_video, conf_v)
        st.success("🎉 영상 처리 완료!")
        st.video(out_path)
        with open(out_path, "rb") as f:
            st.download_button("📥 결과 영상 다운로드", f, file_name="output.mp4")
