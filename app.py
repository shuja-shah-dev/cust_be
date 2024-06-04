from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import time
from yolov8 import YOLOv8
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


model_path = "best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    try:
        uploaded_file = request.files['file']
        image_bytes = uploaded_file.read()
        # Convert bytes to NumPy array
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Perform object detection with YOLOv8
        boxes, scores, class_ids = yolov8_detector(image)

        # Draw detections on the image
        combined_img = yolov8_detector.draw_detections(image)

         
        _, encoded_image = cv2.imencode('.jpg', combined_img)

        return jsonify({
            'image': 'data:image/jpeg;base64,' + base64.b64encode(encoded_image).decode('utf-8'),
        })
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': 'Error processing image'}), 500



if __name__ == "__main__":
    app.run(debug=True, port=5000)

