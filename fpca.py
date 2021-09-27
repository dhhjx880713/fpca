# encoding: UTF-8
import numpy as np
from sklearn.decomposition import PCA
from ...construction.fpca.functional_data import FunctionalData
from ...construction.utils import get_cubic_b_spline_knots
import copy


class FPCA(PCA):
    '''
    Bspline based functional Prinicipal Component Analysis
    '''
    def __init__(self, n_components=None, n_basis=None):
        super(FPCA, self).__init__(n_components)
        self.n_basis = n_basis

    def functional_representation(self, X):
        X = np.asarray(copy.deepcopy(X))
        n_samples, n_features = X.shape
        if self.n_basis is not None:
            X_f = FunctionalData().convert_to_functional_data(X, self.n_basis)
        else:
            self.n_basis = int(0.2 * n_features)
            X_f = FunctionalData().convert_to_functional_data(X, self.n_basis)
        return X_f

    def fit(self, X):
        '''
        fit model with X
        :param X: ndarray, n_samples * n_features
        :return: Return the instance itself

        '''
        X_f = self.functional_representation(X)
        super(FPCA, self).fit(X_f)
        return self

    def transform(self, X):
        X_f = self.functional_representation(X)
        X_transformed = super(FPCA, self).transform(X_f)
        self.n_dims = X.shape[1]
        return X_transformed

    def inverse_transform(self, X):
        X_f_reconstruction = super(FPCA, self).inverse_transform(X)
        knots = get_cubic_b_spline_knots(self.n_basis)
        X_reconstruction = FunctionalData().convert_functional_data_to_discrete_data(X_f_reconstruction, knots,
                                                                                     self.n_dims)
        return X_reconstruction

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)