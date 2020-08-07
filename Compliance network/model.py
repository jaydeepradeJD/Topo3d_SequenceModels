from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
from keras import backend as K
K.set_session(sess)
K.set_image_data_format('channels_last')

import numpy as np
import keras
from tensorflow.keras import activations
from keras.layers import Dense, Input, Multiply, Add, TimeDistributed, LSTM, Activation
from keras.layers import Conv3D, Flatten, concatenate, BatchNormalization, Lambda
from keras.layers import Reshape, Conv3DTranspose, Dropout, SpatialDropout3D, GlobalAveragePooling3D
from keras.models import Model, Sequential
from keras.losses import kld, mse, binary_crossentropy
from keras.utils import plot_model
import tensorflow as tf


def customLoss(y_true, y_pred):
    conf_loss = tf.losses.log_loss(y_true, y_pred, reduction=tf.losses.Reduction.MEAN)
    total_loss = conf_loss 
    return total_loss

class CNN(object):

    def __init__(self):
        pass
    
    def gen_model(self):
        # network parameters
        input_shape = (32, 32, 32, 1)
        current_density = Input(shape=input_shape, name='CurrentDensity')
        Inputs = current_density
        
        model = self.cnn(Inputs)
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=1e-6, amsgrad=False)
        
        model.compile(optimizer=adam, loss='mae', metrics=['mse'])
        print("Compiled model")
        return model
 
        
    def cnn(self, Inputs):
        x = Inputs

        x_3 = Conv3D(filters=16, kernel_size=3, activation='relu', strides=1, padding='same')(x)
        x_5 = Conv3D(filters=16, kernel_size=5, activation='relu', strides=1, padding='same')(x)
        x_7 = Conv3D(filters=16, kernel_size=7, activation='relu', strides=1, padding='same')(x)        
        x = concatenate([x_3, x_5, x_7], axis=-1) 
        x = BatchNormalization()(x)
        
        x = Conv3D(filters=16, kernel_size=3, activation='relu', strides=1, padding='same')(x)
        output = Conv3D(filters=1, kernel_size=3, activation='relu', strides=1, padding='same')(x)
                


        # instantiate model
        return Model(inputs=Inputs, outputs=output, name='CurrentCompliance')
        


    
    
        

