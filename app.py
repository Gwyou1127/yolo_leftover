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

# 경고 메시지 및 로그 최소화
warnings.filterwarnings('ignore')
os.environ['YOLO_VERBOSE'] = 'False'

# 페이지 설정
st.set_page_config(
    page_title="🍴 잔반 탐지기",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/your-username/food-waste-detector',
        'Report a bug': 'https://github.com/your-username/food-waste-detector/issues',
        'About': """
        # 🍴 잔반 탐지기
        
        이 애플리케이션은 YOLOv8 모델을 사용하여 이미지에서 다양한 객체를 감지합니다.
        
        **주요 기능:**
        - 실시간 객체 감지
        - 다중 이미지 처리
        - 신뢰도 조정
        - 감지 통계 제공
        
        **사용 방법:**
        1. 이미지를 업로드하세요
        2. 감지 결과를 확인하세요
        3. 필요시 신뢰도를 조정하세요
        """
    }
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .upload-section {
        border: 3px dashed #ddd;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #007bff;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .detection-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .detection-card:hover {
        transform: translateX(5px);
    }
    
    .confidence-high {
        color: #28a745;
        font-weight: bold;
        background: #d4edda;
        padding: 2px 8px;
        border-radius: 4px;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
        background: #fff3cd;
        padding: 2px 8px;
        border-radius: 4px;
    }
    
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
        background: #f8d7da;
        padding: 2px 8px;
        border-radius: 4px;
    }
    
    .status-connected {
        color: #28a745;
        font-weight: bold;
        background: #d4edda;
        padding: 8px 12px;
        border-radius: 6px;
        display: inline-block;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
        background: #f8d7da;
        padding: 8px 12px;
        border-radius: 6px;
        display: inline-block;
    }
    
    .stats-container {
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #ced4da;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 3rem;
    }
    
    /* 모바일 반응형 */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .upload-section {
            padding: 1rem;
        }
        
        .detection-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# YOLO 모델 로드 (캐싱으로 메모리 효율성 증대)
@st.cache_resource(show_spinner=True)
def load_model():
    """YOLO 모델을 로드하고 캐시에 저장"""
    try:
        with st.spinner("🤖 AI 모델을 로드하는 중..."):
            # CPU 사용 명시적 설정
            model = YOLO('yolov8n.pt')
            model.to('cpu')
            
            # 모델 워밍업 (첫 추론 속도 개선)
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            model(dummy_image, verbose=False)
            
        return model, "✅ 모델 로드 성공"
    except Exception as e:
        st.error(f"모델 로드 중 오류가 발생했습니다: {str(e)}")
        return None, f"❌ 모델 로드 실패: {str(e)}"

# 객체 감지 함수 (최적화된 캐싱)
@st.cache_data(show_spinner=True, max_entries=3)
def detect_objects(image_bytes, confidence_threshold=0.5):
    """이미지에서 객체를 감지 (결과 캐싱)"""
    model, status = load_model()
    
    if model is None:
        return None, [], status
    
    try:
        # bytes를 PIL Image로 변환
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(image)
        
        # 이미지 크기 제한 (메모리 절약)
        max_size = 1024
        height, width = img_array.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_array = cv2.resize(img_array, (new_width, new_height))
        
        # YOLO 추론 (verbose=False로 로그 최소화)
        results = model(img_array, conf=confidence_threshold, verbose=False)
        
        # 결과 처리
        detections = []
        annotated_image = img_array.copy()
        
        # 색상 팔레트 (더 다양한 색상)
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
                for i, box in enumerate(result.boxes):
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # 클래스 정보
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    # 신뢰도가 임계값 이상인 경우만 표시
                    if confidence >= confidence_threshold:
                        # 클래스별 색상 선택
                        color = colors[class_id % len(colors)]
                        
                        # 바운딩 박스 그리기 (두께 증가)
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
                        
                        # 라벨 텍스트 준비
                        label = f"{class_name} {confidence:.2f}"
                        
                        # 라벨 배경 크기 계산
                        font_scale = 0.8
                        thickness = 2
                        (label_width, label_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                        )
                        
                        # 라벨 배경 그리기
                        cv2.rectangle(
                            annotated_image,
                            (x1, y1 - label_height - baseline - 10),
                            (x1 + label_width + 10, y1),
                            color,
                            -1
                        )
                        
                        # 라벨 텍스트 그리기
                        cv2.putText(
                            annotated_image,
                            label,
                            (x1 + 5, y1 - baseline - 5),
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

def get_confidence_color_class(confidence):
    """신뢰도에 따른 CSS 클래스 반환"""
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
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #007bff; margin: 0;">📊 총 객체</h3>
            <p style="font-size: 2rem; font-weight: bold; margin: 0;">{}</p>
        </div>
        """.format(len(detections)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #28a745; margin: 0;">🏷️ 클래스 수</h3>
            <p style="font-size: 2rem; font-weight: bold; margin: 0;">{}</p>
        </div>
        """.format(len(class_counts)), unsafe_allow_html=True)
    
    with col3:
        avg_conf = sum(det['confidence'] for det in detections) / len(detections)
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #ffc107; margin: 0;">🎯 평균 신뢰도</h3>
            <p style="font-size: 2rem; font-weight: bold; margin: 0;">{:.1%}</p>
        </div>
        """.format(avg_conf), unsafe_allow_html=True)
    
    with col4:
        max_conf = max(det['confidence'] for det in detections)
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #dc3545; margin: 0;">⭐ 최고 신뢰도</h3>
            <p style="font-size: 2rem; font-weight: bold; margin: 0;">{:.1%}</p>
        </div>
        """.format(max_conf), unsafe_allow_html=True)
    
    # 클래스별 상세 통계
    if len(class_counts) > 1:
        st.markdown("### 📈 클래스별 감지 현황")
        
        # 감지 수가 많은 순으로 정렬
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        for class_name, count in sorted_classes:
            percentage = (count / len(detections)) * 100
            st.markdown(f"""
            **{class_name}**: {count}개 ({percentage:.1f}%)
            """)

# 메인 앱
def main():
    # 헤더
    st.markdown('<h1 class="main-header">🍴 잔반 탐지기</h1>', unsafe_allow_html=True)
    
    # 앱 소개
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <p style="font-size: 1.2rem; color: #6c757d; margin: 0;">
            🤖 <strong>YOLOv8</strong> 인공지능을 활용한 실시간 객체 감지 시스템
        </p>
        <p style="color: #6c757d; margin: 0.5rem 0 0 0;">
            이미지를 업로드하면 80가지 다양한 객체를 자동으로 감지하고 분석합니다
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 신뢰도 임계값
        confidence_threshold = st.slider(
            "🎯 신뢰도 임계값",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="이 값보다 높은 신뢰도의 객체만 표시됩니다. 값이 높을수록 정확하지만 적은 객체가 감지됩니다."
        )
        
        # 모델 상태 확인
        st.header("📊 시스템 상태")
        model, status = load_model()
        
        if model is not None:
            st.markdown(f'<div class="status-connected">{status}</div>', unsafe_allow_html=True)
            st.info(f"""
            **모델 정보:**
            - 모델: YOLOv8 Nano
            - 감지 가능 클래스: {len(model.names)}개
            - 처리 방식: CPU 최적화
            """)
            
            # 감지 가능한 클래스 목록 (접기/펼치기)
            with st.expander("📋 감지 가능한 객체 목록"):
                classes = list(model.names.values())
                # 3열로 나누어 표시
                cols = st.columns(3)
                for i, class_name in enumerate(classes):
                    with cols[i % 3]:
                        st.write(f"• {class_name}")
        else:
            st.markdown(f'<div class="status-error">{status}</div>', unsafe_allow_html=True)
            st.error("모델 로드에 실패했습니다. 페이지를 새로고침해주세요.")
    
    # 메인 영역
    st.markdown("### 📤 이미지 업로드")
    
    # 파일 업로더
    uploaded_files = st.file_uploader(
        "이미지를 선택하세요",
        type=['jpg', 'jpeg', 'png', 'webp'],
        accept_multiple_files=True,
        help="JPG, PNG, WEBP 형식을 지원합니다. 여러 이미지를 동시에 업로드할 수 있습니다."
    )
    
    if uploaded_files:
        # 업로드된 파일 수에 따라 처리 방식 결정
        if len(uploaded_files) == 1:
            # 단일 이미지 처리
            process_single_image(uploaded_files[0], confidence_threshold)
        else:
            # 다중 이미지 처리
            st.success(f"📁 {len(uploaded_files)}개의 이미지가 업로드되었습니다!")
            
            # 탭으로 구분하여 각 이미지 처리
            tab_names = [f"📸 이미지 {i+1}" for i in range(len(uploaded_files))]
            tabs = st.tabs(tab_names)
            
            for i, (tab, uploaded_file) in enumerate(zip(tabs, uploaded_files)):
                with tab:
                    process_single_image(uploaded_file, confidence_threshold, show_filename=True)
    else:
        # 샘플 이미지 섹션
        display_sample_section(confidence_threshold)
    
    # 푸터
    st.markdown("""
    <div class="footer">
        <p>🔬 Powered by <strong>YOLOv8</strong> & <strong>Streamlit</strong></p>
        <p>Made with ❤️ for AI Object Detection</p>
    </div>
    """, unsafe_allow_html=True)

def process_single_image(uploaded_file, confidence_threshold, show_filename=False):
    """단일 이미지 처리 및 결과 표시"""
    
    # 파일 정보 표시
    if show_filename:
        st.markdown(f"**📄 파일명:** `{uploaded_file.name}`")
    
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.markdown(f"**📊 파일 크기:** {file_size_mb:.2f} MB")
    
    # 파일 크기 제한 확인
    if file_size_mb > 10:
        st.error("❌ 파일 크기가 10MB를 초과합니다. 더 작은 이미지를 업로드해주세요.")
        return
    
    # 이미지 바이트 읽기
    image_bytes = uploaded_file.read()
    
    # 진행 상황 표시
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    progress_text.text("🔍 이미지를 분석하는 중...")
    progress_bar.progress(25)
    
    # 객체 감지 실행
    annotated_image, detections, status = detect_objects(image_bytes, confidence_threshold)
    progress_bar.progress(75)
    
    if annotated_image is not None:
        progress_text.text("✅ 분석 완료!")
        progress_bar.progress(100)
        
        # 잠시 후 진행 상황 제거
        time.sleep(1)
        progress_text.empty()
        progress_bar.empty()
        
        # 결과 표시
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📸 원본 이미지")
            original_image = Image.open(io.BytesIO(image_bytes))
            st.image(original_image, use_container_width=True, caption="업로드된 원본 이미지")
        
        with col2:
            st.markdown(f"#### 🎯 감지 결과 ({len(detections)}개 객체)")
            st.image(annotated_image, use_container_width=True, caption="객체 감지 결과")
        
        # 통계 및 상세 결과 표시
        if detections:
            st.markdown("---")
            st.markdown("### 📊 감지 통계")
            display_detection_stats(detections)
            
            # 상세 결과 리스트
            st.markdown("### 📋 감지된 객체 상세 정보")
            
            # 신뢰도 순으로 정렬
            sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            
            for i, detection in enumerate(sorted_detections, 1):
                confidence = detection['confidence']
                confidence_class = get_confidence_color_class(confidence)
                
                # 객체 크기 계산
                width = detection['bbox'][2] - detection['bbox'][0]
                height = detection['bbox'][3] - detection['bbox'][1]
                
                st.markdown(f"""
                <div class="detection-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0; color: #333;">🏷️ {i}. {detection['class_name']}</h4>
                            <span class="{confidence_class}">{confidence:.1%} 신뢰도</span>
                        </div>
                        <div style="text-align: right; color: #6c757d;">
                            <small>크기: {width} × {height} px</small><br>
                            <small>면적: {detection['area']:,} pixels</small>
                        </div>
                    </div>
                    <div style="margin-top: 10px; color: #6c757d;">
                        <small>📍 위치: ({detection['bbox'][0]}, {detection['bbox'][1]}) → ({detection['bbox'][2]}, {detection['bbox'][3]})</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # 결과 다운로드 옵션
            st.markdown("### 💾 결과 다운로드")
            col1, col2 = st.columns(2)
            
            with col1:
                # 감지 결과 이미지 다운로드
                buf = io.BytesIO()
                annotated_image.save(buf, format='PNG')
                st.download_button(
                    label="🖼️ 감지 결과 이미지 다운로드",
                    data=buf.getvalue(),
                    file_name=f"detected_{uploaded_file.name}",
                    mime="image/png"
                )
            
            with col2:
                # 감지 결과 텍스트 다운로드
                result_text = f"감지 결과 - {uploaded_file.name}\n"
                result_text += f"총 {len(detections)}개 객체 감지\n\n"
                
                for i, det in enumerate(sorted_detections, 1):
                    result_text += f"{i}. {det['class_name']} - {det['confidence']:.2f}\n"
                    result_text += f"   위치: {det['bbox']}\n"
                    result_text += f"   크기: {det['area']} pixels\n\n"
                
                st.download_button(
                    label="📄 감지 결과 텍스트 다운로드",
                    data=result_text,
                    file_name=f"detection_results_{uploaded_file.name}.txt",
                    mime="text/plain"
                )
        else:
            st.warning("🤔 감지된 객체가 없습니다. 신뢰도 임계값을 낮춰보세요.")
            st.info("💡 **팁:** 왼쪽 사이드바에서 신뢰도 임계값을 0.3 이하로 낮춰보세요.")
    else:
        progress_text.text("❌ 분석 실패")
        progress_bar.progress(0)
        st.error(f"이미지 처리에 실패했습니다: {status}")

def display_sample_section(confidence_threshold):
    """샘플 이미지 섹션 표시"""
    st.markdown("### 🎯 샘플 이미지로 테스트해보세요!")
    
    st.info("💡 **사용 팁:** 다양한 객체가 포함된 이미지를 업로드하면 더 정확한 감지 결과를 확인할 수 있습니다.")
    
    # 추천 이미지 타입
    st.markdown("""
    - 🍽️ 음식이 담긴 식판 사진
    """)
    
    # 테스트 가이드
    with st.expander("📖 사용 가이드"):
        st.markdown("""
        1. **이미지 선택**: 위의 파일 업로더를 클릭하여 이미지를 선택하세요
        2. **결과 확인**: 감지된 객체와 통계를 확인하세요
        3. **결과 다운로드**: 필요시 결과 이미지나 텍스트를 다운로드하세요
        
        **최적 결과를 위한 팁:**
        - 📐 이미지 크기: 1024x1024 이하 권장
        - 📊 파일 크기: 10MB 이하
        - 🔍 해상도: 너무 낮지 않은 선명한 이미지
        - 💡 조명: 밝고 선명한 이미지일수록 정확도 향상
        """)

if __name__ == "__main__":
    main()