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
from model import UNET 
from keras.models import load_model, Model
import tensorflow as tf
import keras
from keras import models
from vis_final import vis2d
from vis_final import vis3d
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

    obj = UNET()
    model = obj.gen_model()
    model.load_weights(os.path.join('./models','model.h5'))
        
    cwd = os.getcwd()
    with open('F:\\jrade\\data_32\\change_testData.json', 'r') as f:
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
        final_vd = np.expand_dims(np.expand_dims(smooth(test_data['voxelDensities']), axis=-1), axis=0)
        
        
        for iters in range(2, N_significant_iterations):
            current_se_orig = np.expand_dims(np.expand_dims(smooth(test_data['intermediateSE_%s'%iters]), axis=-1), axis=0)
            # current_vd_orig = np.expand_dims(np.expand_dims(smooth(test_data['intermediateVD_%s'%iters]), axis=-1), axis=0)
            
            current_comp_orig = calcCompliance(current_se_orig, current_vd)
            input_list = [current_comp_orig, current_vd, vf]
            current_vd = model.predict(input_list)
            
        final_vd_pred = current_vd
        loss = np.mean(abs(final_vd - final_vd_pred)) 
        print('L1 loss for sample_%s = '%i, loss)
        path = './preds_final_testdata/Vis2D'        
        vis2d(initial_comp, vd1, final_vd, final_vd_pred, i, path=path)
            
        Inouts = np.ones_like(final_vd)
        path = './preds_final_testdata/Vis_3dslices'
        vis3d(initial_comp[:, :, :, 15:17, :], Inouts[:, :, :, 15:17, :], final_vd[:, :, :, 15:17, :], final_vd_pred[:, :, :, 15:17, :], i, path=path)
            
        path = './preds_final_testdata/Vis3D'
        vis3d(initial_comp, Inouts, final_vd, final_vd_pred, i, path=path)
        
        