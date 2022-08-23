# Import TensorFlow 
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings

warnings.filterwarnings('ignore')
tfds.disable_progress_bar()

# Load the dataset with TensorFlow Datasets. Hint: use tfds.load()
# Create a training set, a validation set and a test set.
(training_set, validation_set, test_set), dataset_info = tfds.load('oxford_flowers102', split = ['train', 'validation', 'test'], as_supervised = True, with_info = True)
image_size = 224

## Explore the Dataset

# print(dataset_info)
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
with open('label_map.json', 'r') as f:
    class_names = json.load(f)
    
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
def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size)) # 224x224 pixels
    image /= 255
    return image, label

batch_size = 64

training_batches = training_set.cache().shuffle(num_training_examples//4).batch(batch_size).map(normalize).prefetch(1)
validation_batches = validation_set.cache().shuffle(num_validation_examples//4).batch(batch_size).map(normalize).prefetch(1)
test_batches = test_set.cache().shuffle(num_test_examples//4).batch(batch_size).map(normalize).prefetch(1)