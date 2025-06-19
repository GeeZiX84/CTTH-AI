import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
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