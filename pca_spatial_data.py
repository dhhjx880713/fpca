# encoding: UTF-8
import numpy as np
import copy
from ...external.PCA import *


class PCASpatialData(object):

    def __init__(self, n_components=None, fraction=0.95):
        self.fraction = fraction
        self.n_components = n_components
        self.pcaobj = None
        self.fileorder = None
        self.centerobj = None
        self.lowVs = None
        self.eigenVecs = None
        self.origin_shape = None

    def fit_motion_dictionary(self, motion_dic):
        self.fileorder = motion_dic.keys()
        self.fit(np.asarray(motion_dic.values()))

    def fit(self, motion_data):
        assert len(motion_data.shape) == 3
        self.origin_shape = motion_data.shape
        motion_data = copy.deepcopy(motion_data)
        reshaped_data = PCASpatialData.reshape_data_for_PCA(motion_data)
        self.centerobj = Center(reshaped_data)
        self.pcaobj = PCA(reshaped_data, fraction=self.fraction)
        if self.n_components is not None:
            print('Number of principal components is: ', self.n_components)
            self.eigenVecs = self.pcaobj.Vt[:self.n_components]
            self.lowVs = np.dot(reshaped_data, self.eigenVecs.T)
        else:
            print('Number of principal components is: ', self.pcaobj.npc)
            self.eigenVecs = self.pcaobj.Vt[:self.pcaobj.npc]
            self.lowVs = np.dot(reshaped_data, self.eigenVecs.T)

    @staticmethod
    def reshape_data_for_PCA(data_mat):
        data_mat = np.asarray(data_mat)
        assert len(data_mat.shape) == 3
        n_samples, n_frames, n_dims = data_mat.shape
        return np.reshape(data_mat, (n_samples, n_frames * n_dims))

    @staticmethod
    def reshape_2D_data_to_motion_data(data_mat_2d, origin_shape):
        assert len(origin_shape) == 3
        data_mat_2d = np.asarray(data_mat_2d)
        n_samples, n_frames, n_dims = origin_shape
        assert n_samples * n_frames * n_dims == data_mat_2d.shape[0] * data_mat_2d.shape[1]
        return np.reshape(data_mat_2d, origin_shape)

    def get_backprojection(self):
        assert self.pcaobj is not None
        backprojection = np.dot(self.lowVs, self.eigenVecs)
        for i in range(len(backprojection)):
            backprojection[i] += self.centerobj.mean
        reconstructed_data = PCASpatialData.reshape_2D_data_to_motion_data(backprojection, self.origin_shape)
        return reconstructed_data