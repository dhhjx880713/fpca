import numpy as np
from . import LEN_QUAT, LEN_EULER, LEN_CARTESIAN


def error_measure_3d_mat(raw_data,
                         reconstructed_data):
    '''
    Compute the mean squared error bewteen original data and reconstructed data
    The data matrix is array3D: n_samples * n_frames * n_dims
    '''
    raw_data = np.asarray(raw_data)
    reconstructed_data = np.asarray(reconstructed_data)
    assert raw_data.shape == reconstructed_data.shape
    diff = raw_data - reconstructed_data
    n_samples, n_frames, n_dim = diff.shape
    err = 0
    for i in range(n_samples):
        for j in range(n_frames):
            err += np.linalg.norm(diff[i, j])
    err = err/(n_samples * n_frames)
    return err



def cartesian_splines_distance(raw_splines, reconstructed_splines, skeleton, weighted_error=True):
    """
    Calculate the Euclidean distance between motion represented as Cartesian splines
    :param raw_splines: Cartesian spline coefficience matrix
    :param reconstructed_splines: spline coefficience matrix
    :param skeleton:
    :param weighted_error:
    :return:
    """
    raw_splines = np.asarray(raw_splines)
    reconstructed_splines = np.asarray(reconstructed_splines)
    n_samples, n_basis, n_dims = raw_splines.shape
    assert n_dims%LEN_CARTESIAN == 0
    n_joints = n_dims/LEN_CARTESIAN
    if not weighted_error:
        return error_measure_3d_mat(raw_splines, reconstructed_splines)/n_joints
    else:
        joint_weights = skeleton.joint_weights[:-4]  # ignore the last two tool joints
        weight_vector = np.ones(n_dims)
        for i in range(n_joints):
            weight_vector[i*LEN_QUAT: (i+1)*LEN_QUAT] *= joint_weights[i]
        weight_mat = np.diag(weight_vector)
        return error_measure_3d_mat(np.dot(raw_splines, weight_mat),
                                    np.dot(reconstructed_splines, weight_mat))/n_joints


def get_cubic_b_spline_knots(n_basis, n_canonical_frames):
    """ create cubic bspline knot list, the order of the spline is 4
    :param n_basis: number of knots
    :param n_canonical_frames: length of discrete samples
    :return:
    """
    n_orders = 4
    knots = np.zeros(n_orders + n_basis)
    # there are two padding at the beginning and at the end
    knots[3: -3] = np.linspace(0, n_canonical_frames-1, n_basis-2)
    knots[-3:] = n_canonical_frames - 1
    return knots