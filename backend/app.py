from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import io
from PIL import Image
import os
import traceback
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import gridfs
import datetime

# ===============================
# 🔗 MongoDB Atlas Connection
# ===============================
MONGO_URI = "mongodb+srv://giri_balan_777:Balan47giri07*@cluster0.89b0jju.mongodb.net/?appName=Cluster0"

client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

# Test connection
try:
    client.admin.command('ping')
    print("✅ MongoDB connection successful!")
except Exception as e:
    print("❌ MongoDB connection failed:", e)

db = client["oral_cancer_db"]
fs = gridfs.GridFS(db)

# ===============================
# 🚀 Flask App Configuration
# ===============================
app = Flask(__name__)

CORS(app, 
     resources={r"/*": {
         "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type"]
     }})

# ===============================
# 🤖 Load TensorFlow Model
# ===============================
MODEL_PATH = "model/oral_cancer_final_model.h5"

try:
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found at {MODEL_PATH}")
        model = None
    else:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
        print(f"📊 Model input shape: {model.input_shape}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    traceback.print_exc()
    model = None

# ===============================
# 🧩 Helper Functions
# ===============================
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ===============================
# 🌐 API Endpoints
# ===============================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "✅ Oral Cancer Detection API is active!",
        "model_loaded": model is not None,
        "mongo_connected": True,
        "endpoints": {
            "predict": "/predict (POST)",
            "history": "/history (GET)",
            "download": "/download/<filename> (GET)"
        }
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "database_connected": True
    }), 200


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """Handle image upload, predict, and store in MongoDB"""
    if request.method == "OPTIONS":
        return "", 200

    try:
        if model is None:
            return jsonify({'error': 'Model not loaded.'}), 500

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400

        print(f"📥 Processing file: {file.filename}")

        # Read image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediction
        prediction = model.predict(img_array, verbose=0)[0][0]
        label = "Cancer Detected 🩸" if prediction > 0.5 else "No Cancer Detected ✅"
        confidence = round(float(prediction if prediction > 0.5 else 1 - prediction) * 100, 2)

        # Save to MongoDB
        fs.put(
            img_bytes,
            filename=file.filename,
            contentType=file.content_type,
            prediction=label,
            confidence=confidence,
            timestamp=datetime.datetime.utcnow()
        )

        result = {
            "prediction": label,
            "confidence": confidence,
            "raw_score": float(prediction)
        }

        print(f"✅ Result: {result}")
        return jsonify(result), 200

    except Exception as e:
        print("❌ Error during prediction:", e)
        traceback.print_exc()
        return jsonify({'error': f"Processing error: {str(e)}"}), 500


@app.route("/history", methods=["GET"])
def history():
    """List previous predictions stored in MongoDB"""
    try:
        files = db.fs.files.find().sort("timestamp", -1)
        results = []
        for f in files:
            results.append({
                "filename": f.get("filename"),
                "prediction": f.get("prediction"),
                "confidence": f.get("confidence"),
                "timestamp": f.get("timestamp")
            })
        return jsonify(results), 200
    except Exception as e:
        print("❌ Error fetching history:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/download/<filename>", methods=["GET"])
def download_image(filename):
    """Download or preview image from MongoDB"""
    try:
        file = fs.find_one({"filename": filename})
        if not file:
            return jsonify({"error": "File not found"}), 404
        
        # Return the image as HTTP response
        return Response(file.read(), mimetype=file.content_type,
                        headers={"Content-Disposition": f"inline; filename={filename}"})
    except Exception as e:
        print("❌ Error retrieving image:", e)
        return jsonify({"error": str(e)}), 500


# ===============================
# ▶️ Run Flask Server
# ===============================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Starting Flask server for Oral Cancer Detection...")
    print("="*50)
    print(f"📁 Model path: {MODEL_PATH}")
    print(f"📁 Model exists: {os.path.exists(MODEL_PATH)}")
    print(f"🔧 Model status: {'✅ Loaded' if model else '❌ Not loaded'}")
    print(f"🌐 Server running at: http://localhost:5000")
    print("="*50 + "\n")

    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
