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


