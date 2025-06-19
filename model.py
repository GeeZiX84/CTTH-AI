import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import kagglehub
image_path = kagglehub.dataset_download("constantinwerner/cyrillic-handwriting-dataset")



def preprocess_image(image_path, target_size=(128, 128)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=1)  # Force grayscale
    image = tf.image.resize(image, target_size)
    image = tf.image.per_image_standardization(image)
    return image

def load_images_and_labels(directory):
    preprocessed_images = []
    labels = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image = preprocess_image(file_path)
                preprocessed_images.append(image)
                label = filename.split("_")[0]  # Adjust based on naming convention
                labels.append(label)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    return np.array(preprocessed_images), np.array(labels)

dataset_path = os.path.join(image_path, "train")
images, labels = load_images_and_labels(dataset_path)

print(f"Sample files in directory: {os.listdir(dataset_path)[:5]}")
print(f"Number of images: {len(images)}")
print(f"Sample labels: {labels[:5]}")

X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Resize the images to match the model's input shape
def resize_images(images, target_size=(64, 64)):
    resized_images = [tf.image.resize(image, target_size) for image in images]
    return np.array(resized_images)

# Resize datasets
X_train = resize_images(X_train, target_size=(64, 64))
X_val = resize_images(X_val, target_size=(64, 64))
X_test = resize_images(X_test, target_size=(64, 64))

# Verify shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")


label_encoder = LabelEncoder()
label_encoder.fit(labels)
y_train = label_encoder.transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)

print(f"Encoded y_train: {y_train[:5]}")
print(f"Number of classes: {len(label_encoder.classes_)}")

m,n = 64

def init_params():
    W1 = np.random.rand(10, 64) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
