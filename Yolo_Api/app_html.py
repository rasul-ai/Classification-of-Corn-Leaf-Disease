from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import cv2
import os
import shutil
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('./model/best.pt')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_first_prediction_txt(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()  # Read the first line
        if first_line:
            confidence_score, class_name = first_line.split()  # Split into confidence and class name
            return {'class_name': class_name, 'confidence_score': float(confidence_score)}  # Return in dictionary
    return None  # Return None if the file is empty

def predict_on_image(image):
    try:
        # Define the folder where the labels (predictions) are stored
        save_dir = os.path.join('runs', 'classify')

        # Clear the previous label files (if any exist)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)  # Remove the directory and its contents

        # Perform classification prediction
        results = model.predict(image, imgsz=768, save_txt=True, save=True)
        save_dir = results[0].save_dir  # Get the new save directory

        # Initialize an empty list to store predictions
        predictions = []

        # Iterate over the saved files in the folder
        for txt_file in os.listdir(os.path.join(save_dir, 'labels')):
            if txt_file.endswith('.txt'):
                file_path = os.path.join(save_dir, 'labels', txt_file)
                first_prediction = read_first_prediction_txt(file_path)
                if first_prediction:
                    predictions.append(first_prediction)

        return predictions  # Return the predictions

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if request.content_type == 'application/json':
            # Handle JSON input
            data = request.json
            if 'image' not in data:
                return jsonify({'error': 'No image data provided'}), 400

            try:
                # Decode the image from base64
                image_data = base64.b64decode(data['image'])
                np_img = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

                if image is None:
                    return jsonify({'error': 'Image decoding failed'}), 400

                predictions = predict_on_image(image)
                if predictions is None:
                    return jsonify({'error': 'Prediction failed'}), 500

                # Convert the image back to base64 (if needed)
                retval, buffer = cv2.imencode('.png', image)
                predicted_img_base64 = base64.b64encode(buffer).decode('utf-8')

                return jsonify({
                    'predicted_image': predicted_img_base64,
                    'predictions': predictions
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        else:
            # Handle form data (HTML)
            if 'file' not in request.files:
                return render_template('index.html', error='No file part')

            file = request.files['file']

            if file.filename == '':
                return render_template('index.html', error='No selected file')

            if file and allowed_file(file.filename):
                image_data = np.asarray(bytearray(file.stream.read()), dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

                if image is None:
                    return render_template('index.html', error='Image decoding failed')

                predictions = predict_on_image(image)
                if predictions is None:
                    return render_template('index.html', error='Prediction failed')

                # Convert image to base64 for HTML display
                retval, buffer = cv2.imencode('.png', image)
                original_img_base64 = base64.b64encode(buffer).decode('utf-8')

                return render_template(
                    'result.html', 
                    original_img_data=original_img_base64, 
                    predictions=predictions
                )

    return render_template('index.html')

if __name__ == '__main__':
    os.environ.setdefault('FLASK_ENV', 'development')
    app.run(debug=False, port=5000, host='0.0.0.0')
