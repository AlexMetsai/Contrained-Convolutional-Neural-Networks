# Alexandros I. Metsai
# alexmetsai@gmail.com
# MIT License

# Test the accuracy of the contrained convolutional neural network.

import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np
import os
import sys
import argparse

# Set argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    '--path',
    type=str,
    help="Path of the network's weights")
    
# Display help if no arguments are given
if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

# Folder path to weights
args = parser.parse_args()
path = args.path

# Define image and batch size
img_height = 256
img_width = 256
batch_size=64

# Load and Compile the model
model = load_model(path)

sgd = SGD(lr=0.001, momentum=0.95, decay=0.0004)

model.compile(
    optimizer=sgd, 
    loss='binary_crossentropy', 
    metrics=['accuracy'])

# Create the Generator
test_data_gen = ImageDataGenerator(preprocessing_function=None,
    rescale=1./255)#RESCALE? 

# Read the positive test samples.
test_generator = test_data_gen.flow_from_directory(
  directory = os.path.join('./test/pos', video_folders[i]),
  target_size=(img_width, img_height), color_mode='grayscale',
  batch_size=batch_size, class_mode="categorical")

# Make predictions for the positive samples.
predictions = model.predict_generator(test_generator)
true_pos = 0
pos_size = len(predictions)

for i in range(pos_size):
    if (predictions > 0.5):
        true_pos += 1

# Read the negative test samples.
test_generator = test_data_gen.flow_from_directory(
  directory = os.path.join('./test/neg', video_folders[i]),
  target_size=(img_width, img_height), color_mode='grayscale',
  batch_size=batch_size, class_mode="categorical")

# Make predictions for the negative samples.
predictions = model.predict_generator(test_generator)
true_neg = 0
neg_size = len(predictions)

for i in range(neg_size):
    if (predictions > 0.5):
        true_neg += 1
