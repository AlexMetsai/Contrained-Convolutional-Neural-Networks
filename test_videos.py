'''
Alexandros I. Metsai
alexmetsai@gmail.com
MIT License

Test the accuracy of the contrained convolutional neural 
network based on predictions of video frame images.
A threshold value is used to make the 
decision for the overall classification.
'''
import os
import sys
import numpy as np
import argparse
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

# Set argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    '--path',
    type=str,
    help="Path of the network's weights")

# Display help if no arguments are given.
if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

# Folder path to weights.
args = parser.parse_args()
path = args.path

# Define image and batch size.
# An error will be displayed if dimensions don't match
# with the ones specified in the training process.
img_height = 256
img_width = 256
batch_size = 64

negative_data_dir = './test/negative'
positive_data_dir = './test/positive'

# Percentage of tampered frames to classify video as fake.
detection_threshold = 0.5

# Load and Compile the model
model = load_model(path)
sgd = SGD(lr=0.001, momentum=0.95, decay=0.0004)
model.compile(
    optimizer=sgd, 
    loss='binary_crossentropy', 
    metrics=['accuracy'])

# Create the Generator
test_data_gen = ImageDataGenerator(preprocessing_function=None,
    rescale=1./255)

# ****************************
# *** Test negative videos ***
# ****************************
video_folders = os.listdir(negative_data_dir)

# Make video-level prediction
correct_guesses = 0

for i in range(len(video_folders)):
    
    print(video_folders[i])
    
    # Read the data from directory
    test_generator = test_data_gen.flow_from_directory(
        directory = os.path.join(negative_data_dir, video_folders[i]),
        target_size=(img_width, img_height), color_mode='grayscale',
        batch_size=batch_size, class_mode="categorical")
    
    # Make predictions for each frame
    predictions = model.predict_generator(test_generator)
    prediction_acc = 0
    for i in range(len(predictions)):
        prediction_acc += (predictions[i]/(len(predictions)))
    
    if (prediction_acc[0] < detection_threshold):
        correct_guesses += 1
    
    print(prediction_acc)

correct_guesses_neg = correct_guesses
negative_len = len(video_folders)

# ****************************
# *** Test positive videos ***
# ****************************
video_folders = os.listdir(positive_data_dir)
correct_guesses = 0
# Make video-level prediction
for i in range(len(video_folders)):
    
    # Read the data from directory
    print(video_folders[i])
    test_generator = test_data_gen.flow_from_directory(
        directory = os.path.join(positive_data_dir, video_folders[i]),
        target_size=(img_width, img_height), color_mode='grayscale',
        batch_size=batch_size, class_mode="categorical")
    
    # Make predictions for each frame
    predictions = model.predict_generator(test_generator)
    prediction_acc = 0
    for i in range(len(predictions)):
        prediction_acc += (predictions[i]/(len(predictions)))
    
    if (prediction_acc[0] > detection_threshold):
        correct_guesses += 1
    
    print(prediction_acc)

print("Correct guesses of real videos", 
  correct_guesses_neg, "out of", negative_len)
print("Percentage of correct guesses for real videos:", 
  correct_guesses_neg/negative_len)
