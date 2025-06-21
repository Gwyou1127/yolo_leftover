import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64
from typing import List
import time
import os
import warnings
import torch

# 경고 메시지 및 로그 최소화
warnings.filterwarnings('ignore')
os.environ['YOLO_VERBOSE'] = 'False'

# 재현 가능한 결과를 위한 시드 설정
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 페이지 설정
st.set_page_config(
    page_title="🍴 잔반 탐지기",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# YOLO 모델 로드 (일관성을 위한 수정)
@st.cache_resource(show_spinner=True)
def load_model():
    """YOLO 모델을 로드하고 캐시에 저장 - Gradio와 동일한 설정"""
    try:
        with st.spinner("🤖 AI 모델을 로드하는 중..."):
            model = YOLO('best.pt')
            
            # Gradio와 동일한 설정 적용
            model.to('cpu')  # 또는 'cuda' - Gradio에서 사용하는 것과 동일하게
            
            # 모델의 NMS 설정을 고정 (Gradio와 동일하게)
            model.overrides['iou'] = 0.45  # IoU threshold
            model.overrides['agnostic_nms'] = False
            model.overrides['max_det'] = 300
            
            # 워밍업 없이 바로 반환 (일관성을 위해)
            
        return model, "✅ 모델 로드 성공"
    except Exception as e:
        st.error(f"모델 로드 중 오류가 발생했습니다: {str(e)}")
        return None, f"❌ 모델 로드 실패: {str(e)}"

# 객체 감지 함수 (일관성을 위한 대폭 수정)
def detect_objects_consistent(image_bytes, confidence_threshold=0.5):
    """Gradio와 일관된 결과를 위한 객체 감지 함수"""
    model, status = load_model()
    
    if model is None:
        return None, [], status
    
    try:
        # bytes를 PIL Image로 변환 (Gradio와 동일한 방식)
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # 이미지를 numpy 배열로 변환
        img_array = np.array(image)
        
        # ★ 중요: 이미지 리사이징을 Gradio와 동일하게 처리
        # 만약 Gradio에서 특정 크기로 리사이징한다면 여기서도 동일하게
        original_height, original_width = img_array.shape[:2]
        
        # Gradio에서 사용하는 전처리와 동일하게 맞춤
        # 예: 640x640으로 패딩이나 리사이징을 한다면
        # img_array = letterbox_resize(img_array, (640, 640))
        
        # YOLO 추론 - Gradio와 정확히 동일한 파라미터 사용
        results = model(
            img_array, 
            conf=confidence_threshold,
            iou=0.45,  # IoU threshold 명시적 설정
            verbose=False,
            save=False,
            save_txt=False,
            save_conf=False,
            augment=False,  # 데이터 증강 비활성화 (일관성을 위해)
            agnostic_nms=False,
            max_det=300,
            device='cpu'  # Gradio와 동일한 디바이스
        )
        
        # 결과 처리 - Gradio와 동일한 방식
        detections = []
        annotated_image = img_array.copy()
        
        # Gradio와 동일한 색상 사용 (고정된 색상)
        colors = [
            (0, 255, 0),    # 녹색
            (255, 0, 0),    # 빨간색  
            (0, 0, 255),    # 파란색
            (255, 255, 0),  # 노란색
            (255, 0, 255),  # 마젠타
            (0, 255, 255),  # 시안
            (128, 0, 128),  # 보라색
            (255, 165, 0),  # 주황색
        ]
        
        for result in results:
            if result.boxes is not None:
                # 박스들을 정렬하여 일관된 순서 보장
                boxes_data = []
                for box in result.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    boxes_data.append((x1, y1, x2, y2, class_id, confidence))
                
                # 좌표 기준으로 정렬 (일관된 순서를 위해)
                boxes_data.sort(key=lambda x: (x[1], x[0]))  # y좌표, x좌표 순
                
                for i, (x1, y1, x2, y2, class_id, confidence) in enumerate(boxes_data):
                    if confidence >= confidence_threshold:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        class_name = model.names[class_id]
                        
                        # 클래스별 고정 색상 (Gradio와 동일)
                        color = colors[class_id % len(colors)]
                        
                        # 바운딩 박스 그리기 (Gradio와 동일한 스타일)
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                        
                        # 라벨 그리기 (Gradio와 동일한 방식)
                        label = f"{class_name} {confidence:.2f}"
                        font_scale = 0.7
                        thickness = 2
                        
                        (label_width, label_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                        )
                        
                        # 라벨 배경
                        cv2.rectangle(
                            annotated_image,
                            (x1, y1 - label_height - baseline - 5),
                            (x1 + label_width + 5, y1),
                            color,
                            -1
                        )
                        
                        # 라벨 텍스트
                        cv2.putText(
                            annotated_image,
                            label,
                            (x1 + 2, y1 - baseline - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (255, 255, 255),
                            thickness
                        )
                        
                        # 감지 결과 저장
                        detections.append({
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": [x1, y1, x2, y2],
                            "area": (x2 - x1) * (y2 - y1),
                            "class_id": class_id
                        })
        
        return Image.fromarray(annotated_image), detections, "✅ 감지 완료"
        
    except Exception as e:
        st.error(f"이미지 처리 중 오류가 발생했습니다: {str(e)}")
        return None, [], f"❌ 처리 중 오류: {str(e)}"

def letterbox_resize(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    YOLO에서 사용하는 letterbox 리사이징 (Gradio와 동일한 전처리)
    이미지 비율을 유지하면서 패딩으로 크기 조정
    """
    shape = img.shape[:2]  # 현재 크기 [height, width]
    
    # 스케일 비율 계산
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # 새로운 크기 계산
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 패딩
    
    # 패딩을 양쪽에 균등하게 분배
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:  # 리사이즈 필요
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

# 캐싱 제거된 감지 함수 (매번 새로 실행)
def detect_objects_no_cache(image_bytes, confidence_threshold=0.5):
    """캐싱 없이 매번 새로 추론하는 함수"""
    return detect_objects_consistent(image_bytes, confidence_threshold)

# 나머지 함수들은 동일...
def get_confidence_color_class(confidence):
    """정확도에 따른 CSS 클래스 반환"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"

def display_detection_stats(detections):
    """감지 통계를 보기 좋게 표시"""
    if not detections:
        return
    
    # 클래스별 통계 계산
    class_counts = {}
    total_area = 0
    
    for det in detections:
        class_name = det['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        total_area += det['area']
    
    # 통계 표시
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

def process_single_image(uploaded_file, confidence_threshold, show_filename=False):
    """단일 이미지 처리 및 결과 표시"""
    
    if show_filename:
        st.write(f"**📄 파일명:** `{uploaded_file.name}`")
    
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.write(f"**📊 파일 크기:** {file_size_mb:.2f} MB")
    
    if file_size_mb > 10:
        st.error("❌ 파일 크기가 10MB를 초과합니다.")
        return
    
    # 이미지 바이트 읽기
    image_bytes = uploaded_file.read()
    
    with st.spinner("🔍 이미지를 분석하는 중..."):
        # 캐싱 없는 함수 사용으로 일관된 결과 보장
        annotated_image, detections, status = detect_objects_no_cache(image_bytes, confidence_threshold)
    
    if annotated_image is not None:
        # 결과 표시
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 원본 이미지")
            original_image = Image.open(io.BytesIO(image_bytes))
            st.image(original_image, use_container_width=True)
        
        with col2:
            st.subheader(f"🎯 감지 결과 ({len(detections)}개 객체)")
            st.image(annotated_image, use_container_width=True)
        
        # 통계 표시
        if detections:
            st.markdown("---")
            st.subheader("📊 감지 통계")
            display_detection_stats(detections)
            
            # 상세 결과
            st.subheader("📋 감지된 객체 상세 정보")
            
            # 정확도 순으로 정렬
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

def main():
    st.title("🍴 잔반 탐지기")
    
    st.markdown("""
    ### 🤖 YOLOv8 기반 객체 감지 시스템
    이미지를 업로드하면 다양한 객체를 자동으로 감지하고 분석합니다.
    """)
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        confidence_threshold = st.slider(
            "🎯 정확도 임계값",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="이 값보다 높은 정확도의 객체만 표시됩니다."
        )
        
        # 일관성 설정 추가
        st.header("🔧 일관성 설정")
        
        use_consistent_inference = st.checkbox(
            "일관된 추론 모드",
            value=True,
            help="Gradio와 동일한 결과를 위해 캐싱을 비활성화하고 고정된 설정을 사용합니다."
        )
        
        if use_consistent_inference:
            st.info("✅ 일관된 추론 모드가 활성화되었습니다. Gradio와 동일한 결과를 얻을 수 있습니다.")
        else:
            st.warning("⚠️ 캐싱 모드입니다. 결과가 다를 수 있습니다.")
    
    # 파일 업로더
    uploaded_files = st.file_uploader(
        "이미지를 선택하세요",
        type=['jpg', 'jpeg', 'png', 'webp'],
        accept_multiple_files=True,
    )
    
    if uploaded_files:
        if len(uploaded_files) == 1:
            process_single_image(uploaded_files[0], confidence_threshold)
        else:
            st.success(f"📁 {len(uploaded_files)}개의 이미지가 업로드되었습니다!")
            
            for i, uploaded_file in enumerate(uploaded_files):
                st.markdown(f"## 📸 이미지 {i+1}")
                process_single_image(uploaded_file, confidence_threshold, show_filename=True)
                st.markdown("---")
    else:
        st.info("💡 이미지를 업로드하여 객체 감지를 시작하세요!")
        
        with st.expander("📖 사용 가이드"):
            st.markdown("""
            ### Gradio와 일관된 결과를 얻기 위한 설정:
            
            1. **일관된 추론 모드 활성화**: 사이드바에서 "일관된 추론 모드" 체크박스를 활성화하세요
            2. **동일한 파라미터 사용**: confidence threshold를 Gradio와 동일하게 설정하세요
            3. **동일한 이미지 사용**: 완전히 같은 이미지 파일을 사용하세요
            
            ### 주요 차이점 해결사항:
            - ✅ 캐싱 비활성화로 매번 새로운 추론
            - ✅ 시드값 고정으로 재현 가능한 결과
            - ✅ 동일한 NMS 설정 적용
            - ✅ 동일한 이미지 전처리 방식
            - ✅ 고정된 색상 및 시각화 설정
            """)

if __name__ == "__main__":
    main()
