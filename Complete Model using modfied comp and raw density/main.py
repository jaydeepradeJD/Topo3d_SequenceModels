''' Code for Keras implementation of MAPDL Topopt Learning
There are three major networks used here, AE, VAE and Unet
The best running models for these are provided in runid 1, 2 and 3 respectively
They correspond to each network architecture mentioned above
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse
import os
    

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=str, default='0', help='GPU option')
args = parser.parse_args()

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
from model_density import UNET as density_model
from model_compliance import CNN as complaince_model
from keras.models import load_model, Model

import keras
from keras import models
from vis import vis2d
from vis import vis3d
from scipy import ndimage
import json
if __name__ == '__main__':

    kernel = np.array([[[1,2,1],[2,4,2],[1,2,1]], [[2,4,2],[4,16,4],[2,4,2]], [[1,2,1],[2,4,2],[1,2,1]]])
    kernel = kernel/np.sum(kernel)  
    def smooth(inputs):
        return ndimage.convolve(inputs, kernel, mode='nearest')
    def calcCompliance(SE, VD):
        SE = SE / np.amax(SE)
        SE = (np.log10(SE) + 22) / 22
        comp = np.multiply(SE, np.power(VD,3))
        return comp

    obj1 = density_model()
    model1 = obj1.gen_model()
    model1.load_weights(os.path.join('./weights/model_density.h5'))
    
    obj2 = complaince_model()
    model2 = obj2.gen_model()
    model2.load_weights(os.path.join('./weights/model_compliance.h5'))
    
    cwd = os.getcwd()
    with open('F:\\jrade\\data_32\\change.json', 'r') as f:
        meta_data = json.load(f)

    for i in range(len(meta_data)):
        SamplePath = meta_data[i]['SamplePath']
        N_significant_iterations = len(meta_data[i]['iterations'])
        
        test_data = np.load(os.path.join(SamplePath, 'voxfields32.npz'))
        vd1 = np.expand_dims(np.expand_dims(smooth(test_data['intermediateVD_1']), axis=-1), axis=0)
        vf = np.ones_like(vd1)*np.mean(vd1)
        
        current_se = np.expand_dims(np.expand_dims(smooth(test_data['intermediateSE_2']), axis=-1), axis=0)
        current_vd = np.expand_dims(np.expand_dims(smooth(test_data['intermediateVD_2']), axis=-1), axis=0)
        initial_comp = calcCompliance(current_se, current_vd)
        current_comp = initial_comp
        final_vd = np.expand_dims(np.expand_dims(smooth(test_data['voxelDensities']), axis=-1), axis=0)
        
        for iters in range(N_significant_iterations):
            input_list_1 = [current_comp, current_vd, vf]
            current_vd = model1.predict(input_list_1)
            input_list_2 = current_vd
            current_comp = model2.predict(input_list_2)

        
        final_vd_pred = current_vd
        loss = np.mean(abs(final_vd - final_vd_pred)) 
        print('L1 loss for sample_%s = '%i, loss)
        # path = './preds_final_alldata/Vis2D'        
        # vis2d(initial_comp, vd1, final_vd, final_vd_pred, i, path=path)
            
        Inouts = np.ones_like(final_vd)
        # path = './preds_final_alldata/Vis_3dslices'
        # vis3d(initial_comp[:, :, :, 15:17, :], Inouts[:, :, :, 15:17, :], final_vd[:, :, :, 15:17, :], final_vd_pred[:, :, :, 15:17, :], i, path=path)
            
        path = './preds_final_alldata/Vis3D'
        vis3d(initial_comp, Inouts, final_vd, final_vd_pred, i, path=path)
        