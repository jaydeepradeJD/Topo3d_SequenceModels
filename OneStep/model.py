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
    vol_loss = tf.square(tf.reduce_mean(y_true - y_pred))
    loss = conf_loss + vol_loss 
    return loss

class UNET(object):

    def __init__(self):
        pass
    def gen_model(self):
        # network parameters
        input_shape = (32, 32, 32, 1)
        initial_compliance = Input(shape=input_shape, name='InitialCompliance')
        vf = Input(shape=input_shape, name='VolumeFraction')
        Inputs = [initial_compliance, vf]
        
        model = self.Unet(Inputs)
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=1e-6, amsgrad=False)
        model.compile(optimizer=adam, loss=customLoss, metrics=['mae', 'mse'])
        print("Compiled model")
        return model
 
        
    def ResNetSEblock(self, inputs, ratio):
        filters = inputs._keras_shape[-1]
        xres = inputs
        x = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)

        x = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        # SE Block Start
        x_se_input = x
        x = GlobalAveragePooling3D()(x)
        x = Dense(filters//ratio, activation='relu')(x)
        x = Dense(filters, activation='sigmoid')(x)
        x = Multiply()([x, x_se_input])
        # SE Block Ends
        x = Add()([x, xres])
        return x

    def Unet(self, Inputs):
        x1, x2 = Inputs[0], Inputs[1]
        x = concatenate([x1, x2], axis=-1)
        
        x = Conv3D(filters=16, kernel_size=3, activation='tanh', strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        l1 = x
        #   16x16
        x = Conv3D(filters=32, kernel_size=3, activation='tanh', strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        l2 = x
        #   8x32
        x = Conv3D(filters=64, kernel_size=3, activation='tanh', strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        #   4x64

        x = self.ResNetSEblock(x, ratio=8)
        x = self.ResNetSEblock(x, ratio=8)
        x = self.ResNetSEblock(x, ratio=8)
        x = self.ResNetSEblock(x, ratio=8)
        x = self.ResNetSEblock(x, ratio=8)
        
        x = Conv3DTranspose(filters=32, kernel_size=3, activation='tanh', strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        #   8x32
        x = concatenate([x, l2])
        #  8x64
        x = Conv3DTranspose(filters=16, kernel_size=3, activation='tanh', strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        # 16x16
        x = concatenate([x, l1])
        # 16x32
        
        x = Conv3DTranspose(filters=8, kernel_size=3, activation='tanh', strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        
        output = Conv3D(filters=1, kernel_size=3, strides=1, activation='sigmoid', padding='same')(x)

        # instantiate model
        return Model(inputs=Inputs, outputs=output, name='FinalDensity')
        


    
    
        

