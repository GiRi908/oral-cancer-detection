from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('../saved_models/oral_cancer_final_model.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        img_file = request.files['file']
        img_path = f'temp_{img_file.filename}'
        img_file.save(img_path)
        img = preprocess_image(img_path)
        pred = model.predict(img)
        result = 'Cancer Detected' if pred > 0.5 else 'Normal'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
