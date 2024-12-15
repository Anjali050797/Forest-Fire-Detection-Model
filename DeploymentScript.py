from flask import Flask, request, render_template_string, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename
import logging

# Initialize Flask app and set up logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load the pretrained model from the specified path
MODEL_PATH = 'C:/forest_fire_data/forestfire_detection_model.h5'
model = load_model(MODEL_PATH)

# Define a simple HTML template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Forest Fire Detection System</title>
</head>
<body>
    <h1>Classify Forest Fire Images</h1>
    <form method="post" action="/predict" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <input type="submit" value="Upload and Predict">
    </form>
    {% if prediction %}
        <h2>Detected: {{ prediction }}</h2>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET'])
def home():
    # Serve the main page
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure there is a file in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400
        
        if file:
            # Make sure the upload directory exists
            upload_folder = os.path.join('c:/forest_fire_data', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            
            # Secure the filename and save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)
            logging.info(f"File saved at {file_path}")
            
            # Prepare the image for prediction
            img = Image.open(file_path)
            img = img.resize((150, 150))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction, axis=1)
            
            # Map prediction index to class name
            class_map = {0: 'Fire', 1: 'No Fire', 2: 'Smoke'}
            predicted_class = class_map.get(predicted_class_index[0], 'Unknown')
            
            return render_template_string(HTML_TEMPLATE, prediction=predicted_class)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
