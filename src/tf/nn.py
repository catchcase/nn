from __future__ import division, print_function, absolute_import

from skimage import color, io
from scipy.misc import imresize
import numpy as np
from sklearn.cross_validation import train_test_split
import os
from glob import glob

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy

import tensorflow as tf

# Import image data, get file paths and sort files
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
files_path = '/PR1/data/train/'
print("Data path: " + files_path)

open_files_path = os.path.join(files_path, 'ope*.png')
closed_files_path = os.path.join(files_path, 'clo*.png')
ope_files = sorted(glob(open_files_path))
clo_files = sorted(glob(closed_files_path))

# Calculate and print total size of dataset
print("'Open' files: " + str(len(ope_files)))
print("'Closed' files: " + str(len(clo_files)))
num_files = len(ope_files) + len(clo_files)
print("Total files: " + str(num_files))

# Shape image data, 64x64 RGB
size_image = 64
aX = np.zeros((num_files, size_image, size_image, 3), dtype='float64')
aY = np.zeros(num_files)
count = 0

# Resize/prepare input data
print("Processing 'open' data")
for f in ope_files:
    # noinspection PyBroadException
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        aX[count] = np.array(new_img)
        aY[count] = 0
        count += 1
    except:
        print("exception occurred; continuing...")
        continue

print("Processing 'closed' data")
for f in clo_files:
    # noinspection PyBroadException
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        aX[count] = np.array(new_img)
        aY[count] = 1
        count += 1
    except:
        print("exception occurred; continuing...")
        continue

# Define test and train datasets: 90% training data, 10% test data
X, X_test, Y, Y_test = train_test_split(aX, aY, test_size=0.1)
Y = to_categorical(Y, 2)
Y_test = to_categorical(Y_test, 2)

# Pre-process the image batch
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Add data augmentation to the image batch
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Input image of 64x64 RGB
network = input_data(shape=[None, 64, 64, 3], data_preprocessing=img_prep, data_augmentation=img_aug)

# 1st Layer - Conv 32 filters (3x3x3)
# 2nd Layer - Max pooling
# 3rd Layer - Conv 64 filters
# 4th Layer - Conv 64 filters
# 5th Layer - Max pooling
# 6th Layer - 512-node FC
# 7th Layer - Fully-connected layer with two outputs

conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')             # 1st Layer
network = max_pool_2d(conv_1, 2)                                               # 2nd Layer
conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')             # 3rd Layer
conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')              # 4th Layer
network = max_pool_2d(conv_3, 2)                                               # 5th Layer
network = fully_connected(network, 512, activation='relu')                     # 6th Layer
network = fully_connected(network, 2, activation='softmax', name='my_output')  # 7th Layer

# Configure training using adam optimizer and categorical cross entropy loss with 0.0005 LR
acc = Accuracy(name="Accuracy")
network = regression(
    network,
    optimizer='adam',
    loss='categorical_crossentropy',
    learning_rate=0.0005, metric=acc)

# Create model object
model = tflearn.DNN(
    network,
    checkpoint_path='model_open_close_6.tflearn',
    max_checkpoints=3,
    tensorboard_verbose=3,
    tensorboard_dir='tmp/tflearn_logs/'
)

# Train for n_epoch epochs
model.fit(
    X,
    Y,
    validation_set=(X_test, Y_test),
    batch_size=5,
    n_epoch=5,
    run_id='model_open_close_6',
    show_metric=True
)

# Remove train ops (for later export as .pb)
with network.graph.as_default():
    del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]

# Save trained model
model.save('model_open_close_6_final.tflearn')
