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
img_height = 256        # CHanged image size!!!
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

# Read the data from the given directory.
test_generator = test_data_gen.flow_from_directory(
  directory = os.path.join('./test/', video_folders[i]),
  target_size=(img_width, img_height), color_mode='grayscale',
  batch_size=batch_size, class_mode="categorical")
