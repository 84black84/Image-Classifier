from PIL import Image
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json

image_size = 224

# initialize and configure the argument parser
p = argparse.ArgumentParser()
p.add_argument("image", action="store", type=str, help="input image path") # required
p.add_argument("model", action="store", type=str, help="model classifier path") # required
p.add_argument("--top_k", action="store", type=str, help="the number of 'k' classes with the hights probability results", default=5) # optional
p.add_argument("--category_names", action="store", type=str, help="JSON file with the label mapping", default=5) # optional
arg_parser = p.parse_args()
    
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

top_classes_number = arg_parser.top_k
model_path = arg_parser.model
image_path = arg_parser.image

# Label Mapping
with open('label_map.json') as json_file:
    class_names = json.load(json_file)

# Load the Keras Model
loaded_keras_model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})
probs, classes = predict(image_path, loaded_keras_model, top_classes_number)
print("\nTop {} classes:\n".format(top_classes_number))
for probability, class_label in zip(probs, classes):
    print("Class -> ", class_label)
    print("Class name -> ", class_names[str(class_label + 1)])
    print("Probability -> {}\n".format(probability))