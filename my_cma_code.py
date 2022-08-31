"""
    This file incluses only the source code needed for answering the questions inside the notebook and         
    helped me during the development phase.
    It can be ignored during the review process.
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from gc import callbacks
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
import time
from PIL import Image
import glob

warnings.filterwarnings('ignore')
tfds.disable_progress_bar()

# Load the dataset with TensorFlow Datasets. Hint: use tfds.load()
# Create a training set, a validation set and a test set.
(training_set, validation_set, test_set), dataset_info = tfds.load('oxford_flowers102', split = ['train', 'validation', 'test'], as_supervised = True, with_info = True)
image_size = 224

## Explore the Dataset

print(dataset_info)
# Get the number of examples in each set from the dataset info.
num_training_examples = dataset_info.splits['train'].num_examples
num_validation_examples = dataset_info.splits['validation'].num_examples
num_test_examples = dataset_info.splits['test'].num_examples
print("There are '{:,}' images in the training set".format(num_training_examples))
print("There are '{:,}' images in the validation set".format(num_validation_examples))
print("There are '{:,}' images in the test set".format(num_test_examples))

# Get the number of classes in the dataset from the dataset info.
num_classes = dataset_info.features['label'].num_classes
print("\nThere are '{:,}' classes in our dataset".format(num_classes))

# Print the shape and corresponding label of 3 images in the training set.
for image, label in training_set.take(3):
    print('\nThe label of the image is: ', label.numpy())
    print('And has:')
    print('\u2022 dtype:', label.dtype) 
    
    print('\nThe image has:')
    print('\u2022 dtype:', image.dtype) 
    print('\u2022 shape:', image.shape)

# Plot 1 image from the training set. 
for image, label in training_set.take(1):
    image = image.numpy().squeeze()
    label = label.numpy()    
    
    # Plot the image
    # Set the title of the plot to the corresponding image label. 
    plt.title(label)
    plt.imshow(image, cmap = plt.cm.binary)
    plt.colorbar()
    plt.show()
    
## Label Mapping
with open('label_map.json') as json_file:
    class_names = json.load(json_file)
    
# print(class_names)
    
# Plot 1 image from the training set. Set the title 
# of the plot to the corresponding class name. 
if str(label) in class_names:
    title = class_names[str(label)]
    plt.title(title)
    plt.imshow(image)
    plt.colorbar()
    plt.show()
    
## Create Pipeline

# Create a pipeline for each set.
#  Cannot batch tensors with different shapes
def apply_transformation(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

def resize_image(image, label):
    image = tf.image.resize(image, (image_size, image_size)) # 224x224 pixels
    return image, label

batch_size = 64

# training_batches = training_set.shuffle(num_training_examples//4).map(apply_transformation).batch(batch_size).prefetch(1)
training_batches = training_set.cache().shuffle(num_training_examples//4).map(resize_image).batch(batch_size).map(apply_transformation).prefetch(1)
validation_batches = validation_set.cache().map(resize_image).batch(batch_size).map(apply_transformation).prefetch(1)
test_batches = test_set.cache().map(resize_image).batch(batch_size).map(apply_transformation).prefetch(1)

# validation_batches = validation_set.map(modify_image).batch(batch_size).prefetch(1)
# test_batches = test_set.map(modify_image).batch(batch_size).prefetch(1)

# Build and Train the Classifier
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL, input_shape=(image_size, image_size,3))

# the weights and biases in our pre-trained model are about to get frozen  
# so that we don't modify them during training
feature_extractor.trainable = False

model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(num_classes, activation = 'softmax')
    ])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS = 100
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(training_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches,
                    callbacks=[early_stopping])

training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range=range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, training_accuracy, label='Training Accuracy')
plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, training_loss, label='Training Loss')
plt.plot(epochs_range, validation_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

## Testing your Network
loss, accuracy = model.evaluate(test_batches)

print('\nLoss on the TEST Set: {:,.3f}'.format(loss))
print('Accuracy on the TEST Set: {:.3%}'.format(accuracy))

## Save the Model
current_time = time.time()
saved_keras_model_filepath = './saved_models/{}.h5'.format(int(current_time))
model.save(saved_keras_model_filepath)

## Load the Keras Model
reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath, custom_objects={"KerasLayer": hub.KerasLayer})
reloaded_keras_model.summary()

# Create the process_image function
def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size)) # 224x224 pixels
    image /= 255
    return image



image_path = './test_images/hard-leaved_pocket_orchid.jpg'
im = Image.open(image_path)
test_image = np.asarray(im)

processed_test_image = process_image(test_image)

fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
ax1.imshow(test_image)
ax1.set_title('Original Image')
ax2.imshow(processed_test_image)
ax2.set_title('Processed Image')
plt.tight_layout()
plt.show()

# Create the predict function
def predict(image_path, model, top_k):
    loaded_image = Image.open(image_path)
    loaded_image = np.asarray(loaded_image)
    processed_image = process_image(loaded_image)
    image_to_predict = np.expand_dims(processed_image, axis=0)
    prediction_results = model.predict(image_to_predict)
    
    values, indices = tf.nn.top_k(prediction_results, k=top_k)
    probs = list(values.numpy()[0])
    classes = list(indices.numpy()[0])
    
    return probs, classes

# image_path = "./test_images/cautleya_spicata.jpg"
image_path = "./test_images/orange_dahlia.jpg"

probs, classes = predict(image_path, model, 5)

print("Probability -> '{}'.".format(probs))
print("Classes -> '{}'.".format(classes))


files = glob.glob("./test_images/*")

# Plot the input image along with the top 5 classes
for image_path in files:
    probs, classes = predict(image_path, model, 5)
    top_classes = [class_names[str(i+1)] for i in classes]

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(Image.open(image_path), cmap = plt.cm.binary)
    ax1.axis('off')
    ax1.set_title(image_path)
    # ax1.set_title(class_names[first_label])
    ax2.barh(np.arange(5), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels(top_classes, size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()