from gc import callbacks
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
import time
from PIL import Image

image_size = 224

# Create the process_image function
def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size)) # 224x224 pixels
    image /= 255
    return image

# Create the predict function
def predict(image_path, model, top_k):
    loaded_image = Image.open(image_path)
    loaded_image = np.asarray(loaded_image)
    processed_image = process_image(loaded_image)
    image_to_predict = np.expand_dims(processed_image, axis=0)
    prediction_results = model.predict(image_to_predict)
    
    prob_res, class_res = tf.nn.top_k(prediction_results, k=top_k)
    probs = list(prob_res.numpy()[0])
    classes = list(class_res.numpy()[0])
    
    return probs, classes