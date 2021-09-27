# -*- coding: utf-8 -*-
"""
Created on Sun Aug 02 21:31:09 2015

"""
import numpy as np
try:
    import rpy2.robjects.numpy2ri as numpy2ri
    import rpy2.robjects as robjects
except:
    print("do not use R code")
from .fpca import FPCA


class FPCATemporalData(object):

    def __init__(self, temporal_data, n_basis, npc):
        """
        *temoral_data: dictionary
        \tDictionary contains filename and its warping index
        """
        self.temporal_data = temporal_data
        self.n_basis = n_basis
        self.npc = npc
        self.z_t_transform_data = {}
        self.temporal_pcaobj = None

    def z_t_transform(self):
        for filename in self.temporal_data:
            tmp = FPCATemporalData._get_monotonic_indices(self.temporal_data[filename])
            assert FPCATemporalData._is_strict_increasing(tmp), \
                ("convert %s to monotonic indices failed" % filename)
            w_tmp = np.array(tmp)
            # add one to each entry, because we start with 0
            w_tmp = w_tmp + 1
            w_tmp = np.insert(w_tmp, 0, 0)  # set w(0) to zero

            w_diff = np.diff(w_tmp)
            z_transform = np.log(w_diff)
            self.z_t_transform_data[filename] = z_transform

    def z_t_transform_vector(self, vec):
        # shift control points to start from 0
        w_tmp = np.array(vec)
        w_tmp -= w_tmp[0]
        w_tmp = self._get_monotonic_indices(w_tmp)
        assert self._is_strict_increasing(w_tmp)

        # add one to each entry, because we start with 0
        w_tmp = w_tmp + 1
        w_tmp = np.insert(w_tmp, 0, 0)  # set w(0) to zero

        w_diff = np.diff(w_tmp)
        z_transform = np.log(w_diff)
        return z_transform

    @classmethod
    def _get_monotonic_indices(cls, indices, epsilon=0.01, delta=0):
        """Return an ajusted set of Frameindices which is strictly monotonic

        Parameters
        ----------
        indices : list
        The Frameindices

        Returns
        -------
        A numpy-Float Array with indices similar to the provided list,
        but enforcing strict monotony
        """
        shifted_indices = np.array(indices, dtype=np.float)
        if shifted_indices[0] == shifted_indices[-1]:
            raise ValueError("First and Last element are equal")

        for i in range(1, len(shifted_indices) - 1):
            if shifted_indices[i] > shifted_indices[i - 1] + delta:
                continue

            while np.allclose(shifted_indices[i], shifted_indices[i - 1]) or \
                    shifted_indices[i] <= shifted_indices[i - 1] + delta:
                shifted_indices[i] = shifted_indices[i] + epsilon

        for i in range(len(indices) - 2, 0, -1):
            if shifted_indices[i] + delta < shifted_indices[i + 1]:
                break

            while np.allclose(shifted_indices[i], shifted_indices[i + 1]) or \
                    shifted_indices[i] + delta >= shifted_indices[i + 1]:
                shifted_indices[i] = shifted_indices[i] - epsilon
        return shifted_indices

    @classmethod
    def _is_strict_increasing(cls, indices):
        """ Check wether the indices are strictly increasing ore not

        Parameters
        ----------
        indices : list
        The Frameindices

        Returns
        -------
        boolean
        """
        for i in range(1, len(indices)):
            if np.allclose(indices[i], indices[i - 1]) or indices[i] < indices[i - 1]:
                return False
        return True

    def fpca_on_temporal_data_using_R(self):
        self.z_t_transform()
        file_order = sorted(self.z_t_transform_data.keys())
        timewarping_data = []
        for filename in file_order:
            timewarping_data.append(self.z_t_transform_data[filename])
        timewarping_data = np.transpose(np.asarray(timewarping_data))
        robjects.conversion.py2ri = numpy2ri.numpy2ri
        r_data = robjects.Matrix(np.array(timewarping_data))
        length = timewarping_data.shape[0]
        max_x = length - 1
        rcode = '''
            library(fda)
            basisobj = create.bspline.basis(c(0,{max_x}),{numknots})
            ys = smooth.basis(argvals=seq(0,{max_x},len={length}),
                              y={data},
                              fdParobj=basisobj)
            pca = pca.fd(ys$fd, nharm={nharm})
            pcaVarmax <- varmx.pca.fd(pca)
            scores = pcaVarmax$scores
        '''.format(data=r_data.r_repr(), max_x=max_x,
                   length=length, numknots=self.n_basis, nharm=self.npc)
        robjects.r(rcode)
        self.temporal_pcaobj = robjects.globalenv['pcaVarmax']
        harms = self.temporal_pcaobj[self.temporal_pcaobj.index('harmonics')]
        mean_fd = self.temporal_pcaobj(self.temporal_pcaobj.names.index('meandf'))
        return {'temporal_eigenvectors': np.asarray(harms[harms.names.index('coefs')]),
                'temporal_parameters': np.asarray(mean_fd[mean_fd.names.index('coefs')]),
                'temporal_mean': np.asarray(mean_fd[mean_fd.names.index('coefs')]),
                'n_dim_temporal': timewarping_data.shape[1],
                'n_basis_temporal': self.n_basis}

    def fpca_on_temporal_data(self):
        '''
        apply functional PCA on dtw results
        :return:
        '''
        self.temporal_pcaobj = FPCA(self.npc, self.n_basis)

        file_order = sorted(self.temporal_data.keys())
        timewarping_data = []
        for filename in file_order:
            timewarping_data.append(self.temporal_data[filename])
        timewarping_data = np.asarray(timewarping_data)

        print('timewarping data shape: ', timewarping_data.shape)
        fdata = self.temporal_pcaobj.functional_representation(timewarping_data)
        z_t_fdata = []
        for vec in fdata:
            z_t_fdata.append(self.z_t_transform_vector(vec))
        z_t_fdata = np.asarray(z_t_fdata)
        lowV_paras = self.temporal_pcaobj.fit_transform(z_t_fdata)
        return {'temporal_eigenvectors': self.temporal_pcaobj.components_.T,
                'temporal_parameters': lowV_paras,
                'temporal_mean': self.temporal_pcaobj.mean_,
                'n_dim_temporal': timewarping_data.shape[1],
                'n_basis_temporal': self.n_basis}