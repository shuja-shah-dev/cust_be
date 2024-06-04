from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import time
from yoloseg import YOLOSeg  
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


model = YOLOSeg("best.onnx", conf_thres=0.5, iou_thres=0.45)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    try:
        file = request.files['file']
        image_data = file.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image_array = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        results = model(image_array)
        processed_image = results[0]

        image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)  
        _, encoded_image = cv2.imencode('.jpg', image_bgr)

        return jsonify({
            'image': 'data:image/jpeg;base64,' + base64.b64encode(encoded_image).decode('utf-8'),
        })
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': 'Error processing image'}), 500



if __name__ == "__main__":
    app.run(debug=True, port=5000)