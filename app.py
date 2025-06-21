import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64
import time
import os
import warnings
import torch
import random
import tempfile

# 경고 제거 및 로그 최소화
warnings.filterwarnings('ignore')
os.environ['YOLO_VERBOSE'] = 'False'

# 시드 고정
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 페이지 설정
st.set_page_config(
    page_title="잔반 탐지기",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_resource(show_spinner=True)
def load_model():
    try:
        with st.spinner("🤖 AI 모델을 로드하는 중..."):
            model = YOLO('best.pt')  # YOLOv8 학습된 모델 경로
        return model, "✅ 모델 로드 성공"
    except Exception as e:
        st.error(f"모델 로드 중 오류가 발생했습니다: {str(e)}")
        return None, f"❌ 모델 로드 실패: {str(e)}"

def detect_objects_consistent(image_bytes, confidence_threshold=0.5):
    model, status = load_model()
    if model is None:
        return None, [], status

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        random.seed(42)
        label_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in model.names]

        image = cv2.imread(tmp_path)
        annotated_image = image.copy()

        outputs = model.predict(source=tmp_path, conf=confidence_threshold, iou=0.5)
        results = outputs[0].cpu().numpy()

        detections = []
        for i, det in enumerate(results.boxes.xyxy):
            x1, y1, x2, y2 = map(int, det)
            label_idx = int(results.boxes.cls[i])
            conf = round(float(results.boxes.conf[i]), 2)
            label = model.names[label_idx]
            box_color = label_colors[label_idx]

            label_text = f"{label} {conf:.2f}"
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color, 20, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_image, (x1, y1 - th - 3), (x1 + tw, y1), box_color, -1)
            cv2.putText(annotated_image, label_text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 8)

            detections.append({
                "class_name": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "area": (x2 - x1) * (y2 - y1),
                "class_id": label_idx
            })

        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_image_rgb), detections, "✅ 감지 완료"

    except Exception as e:
        st.error(f"이미지 처리 중 오류가 발생했습니다: {str(e)}")
        return None, [], f"❌ 처리 중 오류: {str(e)}"

def display_detection_stats(detections):
    if not detections:
        return

    class_counts = {}
    total_area = 0
    for det in detections:
        class_name = det['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        total_area += det['area']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 총 객체", len(detections))
    with col2:
        st.metric("🏷️ 클래스 수", len(class_counts))
    with col3:
        avg_conf = sum(det['confidence'] for det in detections) / len(detections)
        st.metric("🎯 평균 정확도", f"{avg_conf:.1%}")
    with col4:
        max_conf = max(det['confidence'] for det in detections)
        st.metric("⭐ 최고 정확도", f"{max_conf:.1%}")

def process_single_image(uploaded_file, confidence_threshold):
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.write(f"**📊 파일 크기:** {file_size_mb:.2f} MB")
    if file_size_mb > 10:
        st.error("❌ 파일 크기가 10MB를 초과합니다.")
        return

    image_bytes = uploaded_file.read()
    with st.spinner("🔍 이미지를 분석하는 중..."):
        annotated_image, detections, status = detect_objects_consistent(image_bytes, confidence_threshold)

    if annotated_image is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📸 원본 이미지")
            original_image = Image.open(io.BytesIO(image_bytes))
            st.image(original_image, use_container_width=True)
        with col2:
            st.subheader(f"🎯 감지 결과 ({len(detections)}개 객체)")
            st.image(annotated_image, use_container_width=True)

        if detections:
            st.markdown("---")
            st.subheader("📊 감지 통계")
            display_detection_stats(detections)

            st.subheader("📋 감지된 객체 상세 정보")
            sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            for i, detection in enumerate(sorted_detections, 1):
                with st.expander(f"🏷️ {i}. {detection['class_name']} ({detection['confidence']:.1%})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**정확도:** {detection['confidence']:.3f}")
                        st.write(f"**클래스 ID:** {detection['class_id']}")
                    with col2:
                        width = detection['bbox'][2] - detection['bbox'][0]
                        height = detection['bbox'][3] - detection['bbox'][1]
                        st.write(f"**크기:** {width} × {height} px")
                        st.write(f"**면적:** {detection['area']:,} pixels")
                    st.write(f"**위치:** ({detection['bbox'][0]}, {detection['bbox'][1]}) → ({detection['bbox'][2]}, {detection['bbox'][3]})")
        else:
            st.warning("🤔 감지된 객체가 없습니다. 정확도 임계값을 낮춰보세요.")
    else:
        st.error(f"이미지 처리에 실패했습니다: {status}")

def process_video(video_file, confidence_threshold):
    model, status = load_model()
    if model is None:
        st.error(status)
        return

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile_path = tfile.name

    cap = cv2.VideoCapture(tfile_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = tfile_path.replace(".mp4", "_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress = st.progress(0)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=confidence_threshold, iou=0.5)[0]
        for i, det in enumerate(results.boxes.xyxy):
            x1, y1, x2, y2 = map(int, det)
            label_idx = int(results.boxes.cls[i])
            conf = round(float(results.boxes.conf[i]), 2)
            label = model.names[label_idx]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 10)

        out.write(frame)
        frame_count += 1
        progress.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    out.release()

    st.success("✅ 영상 처리 완료!")
    st.video(output_path)

def main():
    st.title("🍴 잔반 탐지기")
    st.markdown("""
    ### 🤖 YOLOv8 기반 객체 감지 시스템
    이미지를 업로드하거나 영상을 분석하여 다양한 객체를 감지합니다.
    """)

    with st.sidebar:
        st.header("⚙️ 설정")
        confidence_threshold = st.slider("🎯 정확도 임계값", 0.1, 1.0, 0.5, 0.05)

    tab1, tab2 = st.tabs(["📷 이미지 분석", "🎥 영상 분석"])

    with tab1:
        st.header("🖼️ 이미지 업로드")
        uploaded_files = st.file_uploader("이미지를 선택하세요", type=['jpg', 'jpeg', 'png', 'webp'], accept_multiple_files=True)

        if uploaded_files:
            if len(uploaded_files) == 1:
                process_single_image(uploaded_files[0], confidence_threshold)
            else:
                st.success(f"📁 {len(uploaded_files)}개의 이미지가 업로드되었습니다!")
                tab_titles = [f"📸 이미지 {i+1}" for i in range(len(uploaded_files))]
                image_tabs = st.tabs(tab_titles)
                for i, (tab, uploaded_file) in enumerate(zip(image_tabs, uploaded_files)):
                    with tab:
                        st.markdown(f"## {tab_titles[i]}")
                        process_single_image(uploaded_file, confidence_threshold)
        else:
            st.info("💡 이미지를 업로드하여 객체 감지를 시작하세요!")

    with tab2:
        st.header("🎥 영상 업로드")
        uploaded_video = st.file_uploader("🎬 분석할 영상 파일을 업로드하세요", type=['mp4', 'avi', 'mov'])

        if uploaded_video is not None:
            st.video(uploaded_video)
            with st.spinner("🔍 영상을 분석하는 중입니다..."):
                process_video(uploaded_video, confidence_threshold)
        else:
            st.info("💡 분석할 영상 파일을 업로드하세요!")

if __name__ == "__main__":
    main()
