#!/usr/bin/env bash

# Alexandros I. Metsai
# alexmetsai@gmail.com
# MIT License

# Test the network for all the trained instances.

# Define the folder containing the weights
FOLDER="saved_model/"

# An already existing folder, where results will be moved to.
LOG_FOLDER="logs/"

# Create an array containing the index of each epoch (i.e. weights.10.h5) 
declare -a MODEL_LIST

for i in `seq 1 9`;
do
    MODEL_LIST[$i-1]="0$i"
done

