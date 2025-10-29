# 🧠 Oral Cancer Detection using Deep Learning

This project uses a Convolutional Neural Network (CNN) model to detect oral cancer from medical images.  
It helps automate early diagnosis by classifying images into **Normal** and **Abnormal (Cancerous)** categories.

---

## 📁 Project Structure
```
OralCancerDetection/
├── dataset/
│ ├── train/
│ │ ├── abnormal/
│ │ └── normal/
│ └── test/
│ ├── abnormal/
│ └── normal/
├── models/
│ └── oral_cancer_model.h5
├── scripts/
│ └── train_model.py
├── app.py # Flask web app (for deployment)
├── requirements.txt # Python dependencies
└── README.md
```

---

## 🚀 How to Run the Project

### 1️⃣ Setup Environment

```bash
git clone https://github.com/GiRi908/oral-cancer-detection.git
cd oral-cancer-detection
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On Mac/Linux
pip install -r requirements.txt

```
### 2️⃣ Prepare Dataset

Download dataset from Kaggle: Oral Cancer Dataset

Unzip it into the dataset/ folder as shown above.
### 3️⃣ Train the Model
```
python scripts/train_model.py
```

This will:

Preprocess images (resize, normalize)

Train CNN model

Save trained weights to models/oral_cancer_model.h5

### 4️⃣ Run the Flask App (Deploy Locally)
```
python app.py
```

Then open your browser at 👉 http://127.0.0.1:5000/
You can upload an image and the model will predict whether it’s Normal or Cancerous.

### 🧩 Model Overview

Framework: TensorFlow / Keras

Architecture: CNN (3 Conv Layers + Dense Layers)

Loss Function: Binary Crossentropy

Optimizer: Adam

Accuracy Achieved: ~95% (after 10 epochs on balanced dataset)

### 📊 Sample Output
Image	Prediction

	Normal

	Cancer
### 🧪 Future Improvements

Use Transfer Learning (e.g., MobileNet, VGG16)

Add Grad-CAM for visual explanations

Deploy model on cloud (Streamlit, Render, or HuggingFace Spaces)

### 👨‍💻 Author

Giribalan K
📧 Email: [giri29012006@gmail.com]
🔗 GitHub: https://github.com/GiRi908

### 📜 License

This project is released under the MIT License.
You’re free to use and modify it with credit.


