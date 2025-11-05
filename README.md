#  Oral Cancer Detection using Deep Learning

This project uses a Convolutional Neural Network (CNN) model to detect oral cancer from medical images.  
It helps automate early diagnosis by classifying images into **Normal** and **Abnormal (Cancerous)** categories.

---

##  Project Structure
```
OralCancerDetection/
│
├── backend/                        
│   ├── app.py                      
│   ├── requirements.txt            
│   ├── model/                      
│   │   └── oral_cancer_model.h5 
│
├── frontend/                       
│   ├── package.json
│   ├── public/
│   └── src/
│       └── App.js
│
├── scripts/                       
│   ├── train_model.py
│   ├── preprocess_data.py
│   └── evaluate_model.py
│
├── dataset/                        
│   ├── train/
│   │   ├── abnormal/
│   │   └── normal/
│   └── test/
│       ├── abnormal/
│       └── normal/
│
├── .gitignore                      
├── README.md                       
└── venv/                           


```

##  How to Run the Project

### 1️.Setup Environment

```bash
git clone https://github.com/GiRi908/oral-cancer-detection.git
cd oral-cancer-detection
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On Mac/Linux
pip install -r requirements.txt

```
### 2️.Prepare Dataset

Download dataset from Kaggle: Oral Cancer Dataset

Unzip it into the dataset/ folder as shown above.
### 3️.Train the Model
```
python scripts/train_model.py
```

This will:

Preprocess images (resize, normalize)

Train CNN model

Save trained weights to models/oral_cancer_model.h5

### 4️.Run the Flask App (Deploy Locally)
```
python app.py
```

Then open your browser at 👉 http://127.0.0.1:5000/
You can upload an image and the model will predict whether it’s Normal or Cancerous.

### Model Overview

Framework: TensorFlow / Keras

Architecture: CNN (3 Conv Layers + Dense Layers)

Loss Function: Binary Crossentropy

Optimizer: Adam

Accuracy Achieved: ~95% (after 10 epochs on balanced dataset)

### Sample Output
Image Prediction Results

### Normal	
<img width="500" height="400" alt="Screenshot 2025-11-05 094119" src="https://github.com/user-attachments/assets/6b7227bb-cd17-4a35-a917-d500b50b9134" />

### Abnormal
<img width="500" height="400" alt="Screenshot 2025-11-05 094139" src="https://github.com/user-attachments/assets/91fe5099-f5e3-4888-ae96-91cf763bb998" />

### Future Improvements

Use Transfer Learning (e.g., MobileNet, VGG16)

Add Grad-CAM for visual explanations

Deploy model on cloud (Streamlit, Render, or HuggingFace Spaces)

### Author

Giribalan K
📧 Email: [giri29012006@gmail.com]
🔗 GitHub: https://github.com/GiRi908

### License

This project is released under the MIT License.
You’re free to use and modify it with credit.


