import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image

# Load pre-trained MobileNetV2 model + higher level layers
model = MobileNetV2(weights='imagenet')

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    preds = model.predict(img_array)
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]  # Get the most likely result
    label, confidence = decoded_preds[1], decoded_preds[2] * 100  # Convert to percentage
    
    return label, confidence

def main():
    image_folder = 'images'
    results = []
    
    if not os.path.exists(image_folder):
        print(f"Folder '{image_folder}' not found.")
        return
    
    image_files = sorted(f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg')))
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        label, confidence = classify_image(img_path)
        result = f"{label} ({confidence:.2f}%)"
        results.append((img_file, result))
    
    for img_file, result in results:
        print(f"{img_file}: {result}")

if __name__ == "__main__":
    main()
