from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import os
from PIL import Image
import gdown
def download_model():
    model_path = "yolov8.pt"
    if not os.path.exists(model_path):
        print("Downloading YOLOv8 model from Google Drive...")
        file_id = "1PNyhvH4XNBTrERjrwccB9CPFJl1TN88z"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
app = Flask(__name__)
download_model()
model = YOLO("best.pt")  # Path to your YOLOv8 model

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)

        # Run prediction
        results = model.predict(source=image_path, conf=0.1, iou=0.7)
        
        # Save the image with bounding boxes
        import cv2
        import numpy as np

        # Convert BGR to RGB
        annotated_image = results[0].plot()  # NumPy array in BGR
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        output_file_path = os.path.join(RESULT_FOLDER, "predicted_" + image.filename)
        Image.fromarray(annotated_image_rgb).save(output_file_path)


        # Get top detection
        top_detection = None
        top_conf = 0.0
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                if conf > top_conf:
                    top_detection = {"label": label, "confidence": round(conf, 2)}
                    top_conf = conf

        return jsonify({
            "prediction": top_detection,
            "image_url": "/" + output_file_path.replace("\\", "/")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
