import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import random

# 모델 로드
model = YOLO('best.pt')  # 경로 수정 가능

# Gradio 방식을 따르기 위한 이미지 전처리
def detect_image_rgb(image, conf_thres, iou_thres):
    # Gradio 입력 이미지는 이미 RGB numpy 배열
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 랜덤 색상 생성 (매번 다르게, Gradio처럼)
    label_colors = [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for _ in model.names
    ]

    results = model.predict(source=img, conf=conf_thres, iou=iou_thres)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    viz = img.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cid = int(classes[i])
        conf = float(confs[i])
        color = label_colors[cid]
        label = f"{model.names[cid]} {conf:.2f}"

        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 20, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(viz, (x1, y1 - th - 3), (x1 + tw, y1), color, -1)
        cv2.putText(viz, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 16)

    return cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)

# 상세 결과 정보 리턴함수
def detect_image_with_info(image, conf_thres, iou_thres):
    annotated = detect_image_rgb(image, conf_thres, iou_thres)
    results = model.predict(source=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                             conf=conf_thres, iou=iou_thres)[0]
    outputs = []
    for box, cid, conf in zip(results.boxes.xyxy.cpu().numpy(),
                               results.boxes.cls.cpu().numpy(),
                               results.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        outputs.append({
            "class_name": model.names[int(cid)],
            "confidence": float(conf),
            "bbox": [x1, y1, x2, y2],
            "area": (x2 - x1) * (y2 - y1),
        })
    return annotated, outputs

# Gradio 인터페이스 정의
image_input = gr.Image(type="numpy", label="Input Image")
conf_input = gr.Slider(0.0, 1.0, 0.75, label="Confidence Threshold")
iou_input = gr.Slider(0.0, 1.0, 0.5, label="IoU Threshold")
image_output = gr.Image(type="numpy", label="Detection Result")
table_output = gr.Dataframe(headers=["class_name", "confidence", "bbox", "area"],
                            label="Detection Details")

demo = gr.Interface(
    fn=detect_image_with_info,
    inputs=[image_input, conf_input, iou_input],
    outputs=[image_output, table_output],
    title="🍴 잔반 탐지기 (Gradio UI)",
    description="YOLOv8 기반 객체 탐지. 이미지 업로드 후 감지 결과와 상세표를 확인할 수 있습니다.",
    examples=[["example1.jpg", 0.75, 0.5]],
    allow_flagging="never",
    layout="vertical",
)

if __name__ == "__main__":
    demo.launch()
