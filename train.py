'''
Alexandros I. Metsai
alexmetsai@gmail.com
MIT License

This version is close to the original method.
Modified version should mention architecture changes.
'''

import keras
from keras.layers import Input, Dense, Conv2D, Conv3D, Reshape, \
    BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, \
    Flatten, Dropout
from keras.models import Model, Sequential
from keras.activations import softmax
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np
from PIL import Image

# The callback to be applied at the end of each iteration. This is 
# used to constrain the layer's weights the same way Bayar and Stamm do
# at their paper. It's the core of the method, and the number one source
# for bugs and logic flaws.
class ConstrainLayer(keras.callbacks.Callback, Model):
    
    # Utilized before each batch

    def on_batch_begin(self, batch, logs={}):
            # Get the weights of the first layer
            all_weights = model.get_weights()
            weights = np.asarray(all_weights[0])
            # Constrain the first layer
            weights = constrainLayer(weights)
            # Return the constrained weights back to the network
            all_weights[0] = weights
            model.set_weights(all_weights)

def constrainLayer(weights):
    
    # Scale by 10k to avoid numerical issues while normalizing
    weights = weights*10000
    
    # Set central values to zero to exlude them from the normalization step
    weights[2,2,:,:]=0

    # Pass the weights 
    filter_1 = weights[:,:,0,0]
    filter_2 = weights[:,:,0,1]
    filter_3 = weights[:,:,0,2]
    
    # Normalize the weights for each filter. 
    # Sum in the 3rd dimension, which contains 25 numbers.
    filter_1 = filter_1.reshape(1,1,1,25)
    filter_1 = filter_1/filter_1.sum(3).reshape(1,1,1,1)
    filter_1[0,0,0,12] = -1
    
    filter_2 = filter_2.reshape(1,1,1,25)
    filter_2 = filter_2/filter_2.sum(3).reshape(1,1,1,1)
    filter_2[0,0,0,12] = -1
    
    filter_3 = filter_3.reshape(1,1,1,25)
    filter_3 = filter_3/filter_3.sum(3).reshape(1,1,1,1)
    filter_3[0,0,0,12] = -1
    
    # Prints are for debug reasons.
    # The sums of all filter weights for a specific filter 
    # should be very close to zero.
    '''
    print(filter_1)
    print(filter_2)
    print(filter_3)
    '''
    '''
    print(filter_1.sum(3).reshape(1,1,1,1))
    print(filter_2.sum(3).reshape(1,1,1,1))
    print(filter_3.sum(3).reshape(1,1,1,1))
    '''
    
    # Reshape to original size. 
    filter_1 = filter_1.reshape(1,1,5,5)
    filter_2 = filter_2.reshape(1,1,5,5)
    filter_3 = filter_3.reshape(1,1,5,5)
    
    # Pass the weights back to the original matrix and return.
    weights[:,:,0,0] = filter_1
    weights[:,:,0,1] = filter_2
    weights[:,:,0,2] = filter_3
    
    return weights

#        -----------------Create the model-----------------

inputs = Input(shape=(256,256,1))
x = Conv2D(3, (5,5), strides=1)(inputs)              # Constrained Conv
                                                     # No activation is performed on this layer,  
                                                     # like it is described in the paper.
x = Conv2D(96, (7,7), strides=2,padding='same')(x)   # Conv 2
x = BatchNormalization()(x)                         
x = Activation('tanh')(x)                           
x = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)

x = Conv2D(64, (5,5), strides=1, padding='same')(x)  # Conv 3
x = BatchNormalization()(x)
x = Activation('tanh')(x)
x = MaxPooling2D(pool_size=(3,3), strides=2,)(x)

x = Conv2D(64, (5,5), strides=1, padding='same')(x)   # Conv 4
x = BatchNormalization()(x)
x = Activation('tanh')(x)
x = MaxPooling2D(pool_size=(3,3), strides=2,)(x)

x = Conv2D(128, (1,1), strides=1, padding='same')(x)  # Conv 5
x = BatchNormalization()(x)
x = Activation('tanh')(x)
x = AveragePooling2D(pool_size=(3,3), strides=2,)(x)

x = Flatten()(x)

x = Dense(100)(x)                            # Fully Connected Layer 1
x = Activation('tanh')(x) 
x = Dense(100)(x)                            # Fully Connected Layer 2
x = Activation('tanh')(x)
x = Dense(2, activation='softmax')(x)        # Fully Connected Layer 3

model = Model(inputs,x)
model.summary()

#         ---------------------------------------------------

# The standart Stohastic Gradient Descent is chosen as the optimizer.
# I think the optimizer is not specified in the paper, but other
# optimizers should be put to test as well, like the Nesterov 
# Accelerated Gradient. The Adam and RMSProp are mostly used for 
# deeper neural networks.
# Categorical Crossentropy is selected as the Loss Function, but it
# would be best to test performance of other ways to calculate the loss
# as well. Momentum, learning rate, and decay values are taken 
# straight from the paper.
sgd = SGD(lr=0.001, momentum=0.95, decay=0.0005)
model.compile(
    optimizer=sgd, 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])

img_height = 256
img_width = 256
batch_size=64

# Load the training and validation datasets
train_data_gen = ImageDataGenerator(preprocessing_function=None,
    rescale=1./255, horizontal_flip=True, vertical_flip=True)

validation_data_gen = ImageDataGenerator(preprocessing_function=None,
    rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    directory=r"./data/train_data/",
    target_size=(img_width, img_height), color_mode='grayscale',
    batch_size=batch_size, class_mode="categorical", shuffle=True)

validation_generator = validation_data_gen.flow_from_directory(
    directory=r"./data/validation_data/",
    target_size=(img_width, img_height), color_mode='grayscale',
    batch_size=batch_size, class_mode="categorical", shuffle=True)

#           ------------ Train the Model ------------
callbacks = [ModelCheckpoint('./saved_model/weights.{epoch:02d}.h5',
    monitor='acc',verbose=1, save_best_only=False,
    period=1), ConstrainLayer()]
