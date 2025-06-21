import streamlit as st
import requests
import io
from PIL import Image
import base64

def call_gradio_api(image_bytes, gradio_url, confidence_threshold=0.5):
    """Gradio API를 호출해서 추론 결과를 받아오는 함수"""
    try:
        # 이미지를 base64로 인코딩
        image_b64 = base64.b64encode(image_bytes).decode()
        
        # Gradio API 호출
        response = requests.post(
            f"{gradio_url}/api/predict",
            json={
                "data": [
                    f"data:image/jpeg;base64,{image_b64}",  # 이미지
                    confidence_threshold  # 신뢰도
                ]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # 결과 이미지 디코딩 (Gradio에서 반환하는 형식에 따라 조정 필요)
            if result['data'] and len(result['data']) > 0:
                result_image_data = result['data'][0]
                
                # base64 디코딩
                if result_image_data.startswith('data:image'):
                    image_data = result_image_data.split(',')[1]
                else:
                    image_data = result_image_data
                
                image_bytes = base64.b64decode(image_data)
                result_image = Image.open(io.BytesIO(image_bytes))
                
                return result_image, "✅ Gradio API 호출 성공"
            
        return None, f"❌ API 호출 실패: {response.status_code}"
        
    except Exception as e:
        return None, f"❌ API 호출 중 오류: {str(e)}"

def process_with_gradio_api(uploaded_file, confidence_threshold, gradio_url):
    """Gradio API를 사용해서 이미지 처리"""
    
    st.write(f"**📄 파일명:** `{uploaded_file.name}`")
    
    # 이미지 바이트 읽기
    image_bytes = uploaded_file.read()
    
    with st.spinner("🔍 Gradio API로 이미지 분석 중..."):
        result_image, status = call_gradio_api(image_bytes, gradio_url, confidence_threshold)
    
    if result_image is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 원본 이미지")
            original_image = Image.open(io.BytesIO(image_bytes))
            st.image(original_image, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Gradio 추론 결과")
            st.image(result_image, use_container_width=True)
        
        st.success(status)
    else:
        st.error(status)

# 메인 앱에서 사용
def main():
    st.title("🍴 잔반 탐지기 (Gradio API 연동)")
    
    # Gradio URL 설정
    gradio_url = st.text_input(
        "Gradio 앱 URL", 
        value="https://your-gradio-app.hf.space",
        help="Gradio 앱의 URL을 입력하세요"
    )
    
    confidence_threshold = st.slider(
        "🎯 정확도 임계값",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    uploaded_file = st.file_uploader(
        "이미지를 선택하세요",
        type=['jpg', 'jpeg', 'png', 'webp']
    )
    
    if uploaded_file and gradio_url:
        process_with_gradio_api(uploaded_file, confidence_threshold, gradio_url)

if __name__ == "__main__":
    main()
