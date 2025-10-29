import os
import cv2

def preprocess_images(folder_path, target_size=(224,224)):
    for category in ['cancer', 'normal']:
        path = os.path.join(folder_path, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, target_size)
            cv2.imwrite(img_path, img)

# Example usage:
preprocess_images('dataset/train')
preprocess_images('dataset/test')
