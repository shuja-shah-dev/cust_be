from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import time
from yoloseg import YOLOSeg  
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# socketio = SocketIO(app)
cap = None
camera_index = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' in request.files:
        file = request.files['file']
        overlap_threshold = float(request.form.get('overlap', 50)) / 100
        conf_threshold = float(request.form.get('conf', 50)) / 100

        model = YOLOSeg("best.onnx", conf_thres=conf_threshold, iou_thres=0.45)
        image_data = file.read()
    
    else:
        return jsonify({'error': 'No image data provided'}), 400

    image_array = np.frombuffer(image_data, np.uint8)
    image_array = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    results = model(image_array)
    _, scores, class_ids, masks = results
    print("Class ids are following")
    print(class_ids)
    processed_image, sum_pixels = model.draw_masks(image_array)  # Your existing image processing logic
    ret, buffer = cv2.imencode('.jpg', processed_image)
    processed_image_data = base64.b64encode(buffer).decode('utf-8')
    if len(class_ids) == 0:
        status='no detection'
        return jsonify({
                'image': 'data:image/jpeg;base64,' + processed_image_data,
                'status': status,
            })
    else:
        if ret:
            sum_pixels_class_1 = np.sum([pix for cls, pix in sum_pixels if cls == 0])
            sum_pixels_class_2 = np.sum([pix for cls, pix in sum_pixels if cls == 1])
            ratio = sum_pixels_class_2 / (sum_pixels_class_1 if sum_pixels_class_1 != 0 else 1)
            overlap = 1 - ratio
            status = "True" if overlap >= overlap_threshold else "False"
            return jsonify({
                'image': 'data:image/jpeg;base64,' + processed_image_data,
                'status': status,
                'overlap': overlap,
                'threshold': overlap_threshold,
                'conf': conf_threshold
            })
        return jsonify({'error': 'Failed to process image'}), 500




if __name__ == "__main__":
    app.run(debug=True, port=5000)