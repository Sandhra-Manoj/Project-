from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

model = load_model('AI detection model1.keras')

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((150, 150)) 
    img = img.convert("RGB") 
    img = np.array(img)
    img = img / 255.0  
    return img


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the POST request
        file = request.files['file']
        if file.filename == '':
            return render_template('result.html', result="No file selected", image_name=None)
        # Create the directory if it doesn't exist
        if not os.path.exists('static/uploads'):
            os.makedirs('static/uploads')
        # Save the file to ./uploads
        img_path = "static/uploads/" + file.filename
        file.save(img_path)
        # Preprocess the image
        img = preprocess_image(img_path)
        # Predict
        prediction = model.predict(np.expand_dims(img, axis=0))
        # Process prediction to get result
        result = "PREDICTION : REAL IMAGE" if prediction[0][0] > 0.5 else "PREDICTION : AI IMAGE"
        return render_template('result.html',result=result,image_name=file.filename)
    

if __name__ == '__main__':
    app.run(debug=True)