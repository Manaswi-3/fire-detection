import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model("models/fire_detection_model.h5")

# Function to preprocess the image
def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to detect fire
def detect_fire(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    if prediction[0] > 0.5:
        return "No Fire Detected!"
    else:
        return "Fire Detected."