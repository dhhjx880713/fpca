# encoding: UTF-8
import numpy as np
from .objective_functions import sfpca_objective_func
from . import LEN_QUAT, LEN_CARTESIAN
from sklearn.decomposition import PCA
from mosi_utils_anim.animation_data import BVHReader, SkeletonBuilder
import time
import os
from scipy.optimize import minimize
from mosi_utils_anim.utilities import write_to_json_file, get_data_analysis_folder, load_json_file
from .prepare_data import scale_root_channels, reshape_data_for_PCA,\
                                                         convert_quat_functional_data_to_cartesian_functional_data, \
                                                         reshape_2D_data_to_motion_data
import copy


class ScaledFunctionalPCA(object):
    """

    """

    def __init__(self,
                 elementary_action,
                 motion_primitive,
                 data_repo,
                 functional_motion_data,
                 npc,
                 skeleton_json,
                 knots,
                 n_joints):
        """

        :param functional_motion_data (numpy.array<3d> n_samples * n_coeffs * n_dims): each dimension of motion data is
        represented as a function, the dimension of motion data is: first three channels are Hips' translation, four
        channels for the orientation of each joint
        :return:
        """
        self.functional_motion_data = copy.deepcopy(functional_motion_data)
        self.motion_primitive = motion_primitive
        self.elementary_action = elementary_action
        self.cartesian_motion_data = convert_quat_functional_data_to_cartesian_functional_data(elementary_action,
                                                                                               motion_primitive,
                                                                                               data_repo,
                                                                                               skeleton_json,
                                                                                               self.functional_motion_data,
                                                                                               knots)
        self.npc = npc
        self.knots = knots
        skeleton_bvh = os.path.join(os.path.dirname(__file__), r'../../../skeleton.bvh')
        bvhreader = BVHReader(skeleton_bvh)
        self.skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
        self.skeleton_json = skeleton_json
        self.data_repo = data_repo
        self.n_joints = n_joints
        self.len_weights = self.n_joints + LEN_CARTESIAN
        self.pca = PCA(n_components=self.npc)

    def heuristic_initialization(self):
        scaled_data, root_scale_vector = scale_root_channels(self.functional_motion_data)
        data = (self.functional_motion_data, self.cartesian_motion_data, self.skeleton, self.npc,
                self.elementary_action, self.motion_primitive, self.data_repo, self.skeleton_json, self.knots)
        unscaled_weights = np.ones(self.len_weights)
        unscaled_error = sfpca_objective_func(unscaled_weights, data)
        root_normalization_weights = np.ones(self.len_weights)
        root_normalization_weights[:LEN_CARTESIAN] = 1.0/np.asarray(root_scale_vector)
        root_normalization_error = sfpca_objective_func(root_normalization_weights, data)
        if unscaled_error > root_normalization_error:
            print("using scaled root translation as initial weights")
            self.initialize_weights(root_normalization_weights)
            print("initial error is: ", root_normalization_error)
        else:
            print("using default initial weights")
            self.initialize_weights(unscaled_weights)
            print("initial error is: ", unscaled_error)

    def initialize_weights(self, weight_vec=None):
        if weight_vec is not None:
            self.weight_vec = weight_vec
        else:
            self.weight_vec = np.ones(self.len_weights)

    def optimize_weights(self):
        data = (self.functional_motion_data, self.cartesian_motion_data, self.skeleton, self.npc, self.elementary_action,
                self.motion_primitive, self.data_repo, self.skeleton_json, self.knots)
        bnds = tuple((0.0001, None) for i in range(len(self.weight_vec)))
        start_time = time.clock()
        print("start to optimize feature weights")
        result = minimize(sfpca_objective_func,
                          self.weight_vec,
                          args=(data,),
                          bounds=bnds,
                          method='L-BFGS-B',
                          options={'maxiter': 1e5})

        running_time = time.clock() - start_time
        print('optimization time: ', running_time)
        print('new weights: ', result.x)
        print(type(result.x))
        try:
            output_data = {'optimization time': running_time,
                           'optimal weights': result.x.tolist()}
            data_analysis_folder = get_data_analysis_folder(self.elementary_action,
                                                            self.motion_primitive,
                                                            self.data_repo)
            output_filename = '_'.join([self.elementary_action,
                                        self.motion_primitive,
                                        str(self.npc) + 'npcs',
                                        'optimized_weights.json'])
            print("optimization is done!")

            write_to_json_file(os.path.join(data_analysis_folder, output_filename), output_data)
        except OSError:
            pass
        return result.x

    def fit(self):
        data_analysis_folder = get_data_analysis_folder(self.elementary_action,
                                                        self.motion_primitive,
                                                        self.data_repo)
        optimized_weights_filename = os.path.join(data_analysis_folder, '_'.join([self.elementary_action,
                                                                                  self.motion_primitive,
                                                                                  str(self.npc) + 'npcs',
                                                                                  'optimized_weights.json']))

        if not os.path.isfile(optimized_weights_filename) or True:
            print("cannot find optimized weights, start optimization")
            self.heuristic_initialization()
            weight_vector = self.optimize_weights()
        else:
            optimal_weights_dic = load_json_file(optimized_weights_filename)
            weight_vector = optimal_weights_dic['optimal weights']
        extended_weights = np.zeros(self.functional_motion_data.shape[-1])
        extended_weights[:LEN_CARTESIAN] = weight_vector[:LEN_CARTESIAN]
        for i in range(self.n_joints):
            extended_weights[LEN_CARTESIAN + i*LEN_QUAT: LEN_CARTESIAN + (i+1)*LEN_QUAT] = weight_vector[LEN_CARTESIAN + i]
        print(extended_weights)
        self.weight_mat = np.diag(extended_weights)
        feature_weighted_functional_coeffs = np.dot(self.functional_motion_data, self.weight_mat)
        self.weight_vec = weight_vector
        self.reshaped_functional_data = reshape_data_for_PCA(feature_weighted_functional_coeffs)
        self.pca.fit(self.reshaped_functional_data)

    def transform(self):
        return self.pca.transform(self.reshaped_functional_data)

    def inverse_transform(self, X):
        backprojection = self.pca.inverse_transform(X)
        feature_weighted_functional_data = reshape_2D_data_to_motion_data(backprojection, self.functional_motion_data.shape)
        inv_mat = np.linalg.inv(self.weight_mat)
        functional_data = np.dot(feature_weighted_functional_data, inv_mat)
        return functional_data