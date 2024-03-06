import os
import pandas as pd
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from io import BytesIO
import numpy as np
import base64

app = Flask(__name__, template_folder='.')

model = load_model('best_model_finetuned.h5')

diagnosis_dict = {
    0: 'MEL',
    1: 'NV',
    2: 'BCC',
    3: 'AK',
    4: 'BKL',
    5: 'DF',
    6: 'VASC',
    7: 'SCC',
    8: 'UNK'
}

def load_actual_diagnoses():
    return pd.read_csv('ISIC_2019_Training_GroundTruth.csv')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        try:
            img_bytes = BytesIO(file.read())
            img_bytes_for_display = img_bytes.getvalue() 
            filename = os.path.splitext(file.filename)[0] 
            img = Image.open(img_bytes)
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)
            predicted_diagnosis = diagnosis_dict[predicted_class[0]]
            confidence = np.max(predictions) * 100  

            actual_diagnoses_df = load_actual_diagnoses()
            actual_diagnosis_row = actual_diagnoses_df[actual_diagnoses_df['image'] == filename]
            actual_diagnosis = 'Not available'
            if not actual_diagnosis_row.empty:
                actual_diagnosis = actual_diagnosis_row.iloc[0][1:].idxmax()  

        
            encoded_img = base64.b64encode(img_bytes_for_display).decode('utf-8')
            image_data = f"data:image/jpeg;base64,{encoded_img}"

            return render_template('prediction.html', predicted_diagnosis=predicted_diagnosis, actual_diagnosis=actual_diagnosis, confidence=confidence, image_data=image_data, filename=filename)
        except Exception as e:
            return f"An error occurred: {e}"

    return "Invalid request"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']

if __name__ == '__main__':
    app.run(debug=True)
