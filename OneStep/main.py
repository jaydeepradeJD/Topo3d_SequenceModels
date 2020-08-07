''' Code for Keras implementation of MAPDL Topopt Learning
There are three major networks used here, AE, VAE and Unet
The best running models for these are provided in runid 1, 2 and 3 respectively
They correspond to each network architecture mentioned above
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse
import os


def train_model(MLTOP, b_size, eps, data_files, model_name):

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/', batch_size=b_size, write_graph=True, write_grads=True)
    mcCallBack = keras.callbacks.ModelCheckpoint( os.path.join('./models',model_name), monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=True, mode='auto', period=1)
    rLrCallBack = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)
    # ESCallBack = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=False)


    # train the network
    # for file in data_files[:-1]:
    train_data = DataProcessor(data_files[0], batch_size = b_size, train=True)
    val_data = DataProcessor(data_files[1], batch_size = b_size, train=False)
    MLTOP.fit_generator(generator = train_data, epochs = eps, verbose=1, 
                        validation_data=val_data, callbacks=[tbCallBack, mcCallBack, rLrCallBack])

    metrics = MLTOP.evaluate_generator(val_data)
    print('test metrics', metrics)
    

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--mode", type=str, default='train', help='train and test mode')
parser.add_argument("-g", "--gpu", type=str, default='0', help='GPU option')
parser.add_argument("-b", "--batch_size", type=int, default=64, help='batch size')
parser.add_argument("-e", "--epochs", type=int, default=50, help='num epochs')

args = parser.parse_args()

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from data import DataProcessor
from model import UNET
from keras.models import load_model, Model

from keras.utils import plot_model

import keras
from keras import models
from vis import vis2d
from vis import vis3d

if __name__ == '__main__':

    for path in ['./preds', './logs', './models']:
        if not os.path.exists(path):
            os.makedirs(path)
    
    if args.mode == 'train':
        data_files = ['F:\\jrade\\data_32\\traindata.npz', 'F:\\jrade\\data_32\\valdata.npz'] 
        obj = UNET()
        model = obj.gen_model()
        model.summary()
        plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
        model_name = 'model.h5'
        
        train_model(model, args.batch_size, args.epochs, data_files, model_name=model_name)
    

    if args.mode == 'test':
        #MLTOP.compile(optimizer='adam',loss='mse')
        obj = UNET()
        model = obj.gen_model()
        model.load_weights(os.path.join('./models','model.h5'))
        test_data = DataProcessor('F:\\jrade\\data_32\\testdata.npz',
            batch_size =args.batch_size, train=False)
        # test_data = DataProcessor('F:\\jrade\\data_32\\testcases.npz',
        #     batch_size =args.batch_size, train=False)
        
        k = 0
        print(len(test_data))
        for i in range(len(test_data)):
        # for i in range(1): 
            test_data.labelled = True
            input_list, output = test_data.__getitem__(k)
            
            metrics = model.evaluate(input_list, output)
            print('test metrics', metrics)
            predoutputs = model.predict(input_list)
            
            path = './preds/Vis2D'        
            vis2d(input_list[0], input_list[1], output, predoutputs, k, path=path)
            
            Inouts = np.ones_like(output)
            path = './preds/Vis3D'
            vis3d(input_list[0], Inouts, output, predoutputs, k, path=path)
            
            path = './preds/Vis_3dslices'
            vis3d(input_list[0][:, :, :, 15:17, :], Inouts[:, :, :, 15:17, :], output[:, :, :, 15:17, :], predoutputs[:, :, :, 15:17, :], k, path=path)
            
            k = k+1
    