import numpy as np



def reshape_data_for_PCA(data_mat):
    data_mat = np.asarray(data_mat)
    assert len(data_mat.shape) == 3
    n_samples, n_frames, n_dims = data_mat.shape
    return np.reshape(data_mat, (n_samples, n_frames * n_dims))


def reshape_2D_data_to_motion_data(data_mat_2d, origin_shape):
    assert len(origin_shape) == 3
    data_mat_2d = np.asarray(data_mat_2d)
    n_samples, n_frames, n_dims = origin_shape
    assert n_samples * n_frames * n_dims == data_mat_2d.shape[0] * data_mat_2d.shape[1]
    return np.reshape(data_mat_2d, origin_shape)


def get_semantic_motion_primitive_path(elementary_action,
                                       motion_primitive,
                                       datarepo_dir=None):
    if datarepo_dir is None:
        datarepo_dir = r'E:\workspace\repo'
    # map the old motion primitive name to the new name
    if motion_primitive == 'first':
        motion_primitive = 'reach'
    if motion_primitive == 'second':
        motion_primitive = 'retrieve'
    return os.path.join(datarepo_dir,
                        r'data\3 - Motion primitives\motion_primitives_quaternion_PCA95_temporal_semantic',
                        'elementary_action_' + elementary_action,
                        '_'.join([elementary_action,
                                  motion_primitive,
                                  'quaternion',
                                  'mm.json']))


def convert_quat_functional_data_to_cartesian_functional_data(elementary_action,
                                                              motion_primitive,
                                                              data_repo,
                                                              skeleton_json,
                                                              quat_coeffs_mat,
                                                              knots):
    """
    convert a set of quaternion splines to cartesian splines
    :param quat_coeffs_mat (numpy.array<3d>): n_samples * n_basis * n_dims
    :return cartesian_coeffs_mat (numpy.array<3d>): n_samples * n_basis * n_dims
    """
    semantic_motion_primitive_file = get_semantic_motion_primitive_path(elementary_action,
                                                                        motion_primitive,
                                                                        data_repo)
    quat_spline_constructor = QuatSplineConstructor(semantic_motion_primitive_file,
                                                    skeleton_json)
    quat_coeffs_mat = np.asarray(quat_coeffs_mat)
    n_samples, n_basis, n_dims = quat_coeffs_mat.shape
    cartesian_coeffs_mat = []
    for i in range(n_samples):
        quat_spline = quat_spline_constructor.create_quat_spline_from_functional_data(quat_coeffs_mat[i],
                                                                                      knots)
        cartesian_spline = quat_spline.to_cartesian()
        cartesian_coeffs_mat.append(cartesian_spline.coeffs)
    return np.asarray(cartesian_coeffs_mat)