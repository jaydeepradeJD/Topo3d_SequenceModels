import numpy as np
np.random.seed(123)
import os
import keras
import tensorflow as tf
from scipy import ndimage

class DataProcessor(keras.utils.Sequence):
    """
    'Generator for data'
    """
    def __init__(self, filepath, batch_size=64, train = False):
        self.data = np.load(filepath)
        self.train = train
        
        self.currentSE = np.expand_dims(self.data['currentSE_smooth'],axis=-1)
        self.currentVD = np.expand_dims(self.data['currentVD_smooth'],axis=-1)
        
        self.batch_size = batch_size
        self.labelled = True   
        self.indexes = np.arange(self.currentVD.shape[0])
        
        if self.train:
            np.random.shuffle(self.indexes)
        

    def __len__(self):
        """'Denotes the number of batches per epoch'"""
        return int(len(self.indexes) / self.batch_size)

    def __getitem__(self, index):
        """'Generate one batch of data'"""
        # Generate indexes of the batch
        if index == 'all':
            indexes = self.indexes
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]


        # Generate data
        if self.labelled == True:
            X, y = self._data_process(indexes)
            return X, y
        else:
            X = self._data_process(indexes)
            return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #if self.train == True:
        np.random.shuffle(self.indexes)


    def _calcCompliance(self, SE, VD):
        SE = SE / np.amax(SE)
        SE = (np.log10(SE) + 22) / 22
        comp = np.multiply(SE, np.power(VD,3))
        return comp

    def _data_process(self, indexes):
        def _rot90z(model):
            model = np.rot90(model, k=1, axes=(0, 2))  # rotate around z 90 deg
            return model


        def _rot90y(model):
            model = np.rot90(model, k=1, axes=(0, 1))  # rotate around y 90 deg
            return model


        def _rot90x(model):
            model = np.rot90(model, k=1, axes=(1, 2))  # rotate around x 90 deg
            return model

        def _rot270z(model):
            model = np.rot90(model, k=3, axes=(0, 2))  # rotate around z 90 deg
            return model


        def _rot270y(model):
            model = np.rot90(model, k=3, axes=(0, 1))  # rotate around y 90 deg
            return model


        def _rot270x(model):
            model = np.rot90(model, k=3, axes=(1, 2))  # rotate around x 90 deg
            return model


        def _mirror_yz(model):
            model = np.flip(model, 0)  # mirror y-z plane
            return model


        def _mirror_xy(model):
            model = np.flip(model, 1)  # mirror x-y plane
            return model


        def _mirror_xz(model):
            model = np.flip(model, 2)  # mirror x-z plane
            return model


        def _no_rot(model):  # no rotation
            return model

        action_dict = {1:_rot90x, 2:_rot90y, 3:_rot90z,
                        4:_rot270x, 5:_rot270y, 6:_rot270z,
                        7:_mirror_xz, 8:_mirror_xy, 9:_mirror_yz,
                        10:_no_rot}
        """'Generates data containing batch_size samples'"""
        # X : (n_samples, *dim, n_channels)
        # Y : (n_samples, *dim, n_channels)
        X1 = np.zeros((len(indexes), 32, 32, 32, 1))
        y = np.zeros((len(indexes), 32, 32, 32, 1))
        
        for i, index in enumerate(indexes):
            if self.train:
                action = np.random.randint(1, 10)
            else:
                # shutting off data aug for Testcases but for testdata use data aug for testing
                action = 10 #np.random.randint(1, 10) 
                
                
            cSE = self.currentSE[index, :, :, :, 0]
            cVD = self.currentVD[index, :, :, :, 0]
            
            X1[i, :, :, :, 0] = action_dict[action](cVD)
            
            y[i, :, :, :, 0] = action_dict[action](self._calcCompliance(cSE, cVD))
            
        if self.labelled == True:
                    
            return X1, y
        else:
            return X1