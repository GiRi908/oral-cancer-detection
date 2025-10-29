from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Load model
model = load_model('saved_models/oral_cancer_model.h5')

# Test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Predict
preds = model.predict(test_generator)
pred_labels = (preds > 0.5).astype(int)

print(confusion_matrix(test_generator.classes, pred_labels))
print(classification_report(test_generator.classes, pred_labels))
