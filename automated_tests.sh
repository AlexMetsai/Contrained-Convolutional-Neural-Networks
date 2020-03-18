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

for i in `seq 10 50`;
do
    MODEL_LIST[$i-1]="$i"
done

# NOT FILLED: test the model for all epochs
# Somehow I lost the temp bash script I was using,  
# need to re-write some minor loop code.

# Test the network for all epochs
for i in "${MODEL_LIST[@]}"
do  #testing
    
    # Location of the trained model weights.
    FILE="weights.$i.h5"
    
    # Specify the file to store the output to
    LOG="$FOLDERlog_$FILE.txt"
    
    # Execute and output to log file.
    
