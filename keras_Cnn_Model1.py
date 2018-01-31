#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:21:45 2018

@author: virajdeshwal
"""

''''OkAY so here we are revising keras once again.
we will start by practicising of compiling a MLP model and then we will proceed further.
Once we will be done with the practcising the compilation of the MLP model.
We will train CNN model and we will compare the parameters in MLP and CNN model '''

#intake = input('Press any key to dive into CNN model designing....\n')
'''
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D,Activation,Dense, Flatten


model_mlp = Sequential()
model_mlp.add(Conv2D(filters=16, kernel_size=2, strides=2, padding='same', activation='relu',
                 input_shape=(32,32,3)))
model_mlp.add(MaxPool2D(pool_size=2, strides=1))
model_mlp.add(Conv2D(filters=32, kernel_size=2, strides=2, padding='same', activation='relu'))
model_mlp.add(MaxPool2D(pool_size=2, strides=1))
model_mlp.add(Conv2D(filters=64, kernel_size=2, strides=2,padding='same', activation='relu'))
model_mlp.add(MaxPool2D(pool_size=2, strides=1))
model_mlp.add(Conv2D(filters=128, kernel_size=2,strides=2,padding='same', activation='relu'))
model_mlp.add(MaxPool2D(pool_size=2, strides=2))
model_mlp.add(Flatten())
model_mlp.add(Dense(500, activation='relu'))
model_mlp.add(Dense(10, activation='softmax'))
model_mlp.summary()

model_mlp.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
'''
print('Here we Go!!!\nMLP model is all set.')
print('Now Lets check what we have in CNN model. And compare the parameters.')

from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout,Activation,Flatten

model_cnn = Sequential()
model_cnn.add(Convolution2D(filters=16, kernel_size=2, strides=2, padding='same', activation='relu',
                 input_shape=(32,32,3)))
model_cnn.add(MaxPooling2D(pool_size=2, strides=1))
model_cnn.add(Convolution2D(filters=32, kernel_size=2, strides=2, padding='same', activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=2, strides=1))
model_cnn.add(Convolution2D(filters=64, kernel_size=2, strides=2,padding='same', activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=2, strides=1))
model_cnn.add(Convolution2D(filters=128, kernel_size=2,strides=2,padding='same', activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=2, strides=2))
model_cnn.add(Flatten())
model_cnn.add(Dropout(0.3))
model_cnn.add(Dense(500, activation='relu'))
model_cnn.add(Dropout(0.4))

model_cnn.add(Dense(10, activation='softmax'))
model_cnn.summary()
