import argparse

import cv2
import gradio as gr
from PIL import Image

from traffic_signs.inference import TrafficSignSystem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo", default="outputs/yolo/train/weights/best.pt")
    p.add_argument("--cnn", default="outputs/cnn/best.pt")
    p.add_argument("--share", action="store_true")
    a = p.parse_args()
    system = TrafficSignSystem(a.yolo, a.cnn)

    def predict(image):
        if image is None:
            return None, []
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annotated, predictions = system.predict(bgr)
        result = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        rows = [[p["class_name"], round(p["detection_confidence"], 3),
                 round(p["classification_confidence"], 3)] for p in predictions]
        return Image.fromarray(result), rows

    with gr.Blocks(title="Traffic Sign Recognition") as app:
        gr.Markdown("# Traffic Sign Recognition\nUpload a road image. YOLO locates signs; the CNN names each crop.")
        image = gr.Image(type="numpy", label="Road image")
        button = gr.Button("Detect and classify")
        output = gr.Image(label="Annotated image")
        table = gr.Dataframe(headers=["Sign", "Detection confidence", "CNN confidence"],
                             datatype=["str", "number", "number"], label="Predictions")
        button.click(predict, inputs=image, outputs=[output, table])
    app.launch(share=a.share)


if __name__ == "__main__":
    main()
