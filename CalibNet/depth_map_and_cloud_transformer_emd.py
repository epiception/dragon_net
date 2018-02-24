import tensorflow as tf
import numpy as np
import model_utils

IMG_HT = 375
IMG_WDT = 1242
batch_size = 1

shape = (IMG_HT, IMG_WDT)

# fx = 7.183351e+02
# fy = 7.183351e+02
# cx = 6.003891e+02
# cy = 1.815122e+02
#
# K =  np.array([7.183351e+02, 0.000000e+00, 6.003891e+02, 0.000000e+00, 7.183351e+02, 1.815122e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00]).reshape(3,3)

fx = 7.215377e+02
fy = 7.215377e+02
cx = 6.095593e+02
cy = 1.728540e+02

K = np.array([7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00]).reshape(3,3)

fx_scaled = 2*(fx)/1242.0 #focal length x scaled for -1 to 1 range
fy_scaled = 2*(fy)/375.0 #focal length y scaled for -1 to 1 range
cx_scaled = -1 + 2*(cx - 1.0)/1242.0 # optical center x scaled for -1 to 1 range
cy_scaled = -1 + 2*(cy - 1.0)/375.0 # optical center y scaled for -1 to 1 range

K_mat_scaled = np.array([[fx_scaled,  0.0, cx_scaled],
                         [0.0, fy_scaled,  cy_scaled],
                         [0.0, 0.0, 1.0]], dtype = np.float32)

tf_K_mat = tf.constant(K_mat_scaled, dtype = tf.float32)

depth_map_var = tf.placeholder(tf.float32, shape = [IMG_HT, IMG_WDT])
transform_mat_var = tf.placeholder(tf.float32, shape = [3, 4])

small_transform = tf.constant(np.array([1.0, 0.0, 0.0, (4.485728e+01)/fx,
                                        0.0, 1.0, 0.0, (2.163791e-01)/fy,
                                        0.0, 0.0, 1.0, 2.745884e-03,
                                        0.0, 0.0, 0.0, 1.0]).reshape(4,4), dtype = tf.float32)

def sparsify_cloud(S):
    with tf.device('/cpu:0'):
        point_limit = 4096
        no_points = tf.shape(S)[0]
        no_partitions = no_points/tf.constant(point_limit, dtype=tf.int32)
        saved_points = tf.gather_nd(S, [tf.expand_dims(tf.range(0, no_partitions*point_limit), 1)])
        saved_points = tf.reshape(saved_points, [point_limit, no_partitions, 3])
        saved_points_sparse = tf.reduce_mean(saved_points, 1)

        return saved_points_sparse



def _cloud_transformer(depth_map, t_mat):
    transformed_points = _3D_meshgrid_batchwise_diff(IMG_HT, IMG_WDT, depth_map, batch_size, transformation_matrix=t_mat)

    return transformed_points

def _3D_meshgrid_batchwise_diff(height, width, depth_img, num_batch, transformation_matrix):

    # # Creates 3d sampling meshgrid

    x_index = tf.linspace(-1.0, 1.0, width)
    y_index = tf.linspace(-1.0, 1.0, height)
    z_index = tf.range(0, width*height)

    x_t, y_t = tf.meshgrid(x_index, y_index)

    # flatten
    x_t_flat = tf.reshape(x_t, [1,-1])
    y_t_flat = tf.reshape(y_t, [1,-1])
    ZZ = tf.reshape(depth_img, [-1])

    zeros_target = tf.zeros_like(ZZ)
    mask = tf.not_equal(ZZ, zeros_target)
    ones = tf.ones_like(x_t_flat)

    sampling_grid_2d = tf.concat([x_t_flat, y_t_flat, ones], 0)
    sampling_grid_2d_sparse = tf.transpose(tf.boolean_mask(tf.transpose(sampling_grid_2d), mask))
    ZZ_saved = tf.boolean_mask(ZZ, mask)
    ones_saved = tf.expand_dims(tf.ones_like(ZZ_saved), 0)

    projection_grid_3d = tf.matmul(tf.matrix_inverse(tf_K_mat), sampling_grid_2d_sparse*ZZ_saved)
    homog_points_3d = tf.concat([projection_grid_3d, ones_saved], 0)

    final_transformation_matrix_pred = tf.matmul(transformation_matrix, small_transform)[:3,:]
    warped_sampling_grid_pred = tf.matmul(final_transformation_matrix_pred, homog_points_3d)

    points_2d_pred = tf.matmul(tf_K_mat, warped_sampling_grid_pred[:3,:])
    Z_pred = points_2d_pred[2,:]

    x_dash_pred = points_2d_pred[0,:]
    y_dash_pred = points_2d_pred[1,:]
    warped_sampling_grid_remake_pred = tf.stack([x_dash_pred, y_dash_pred, Z_pred], 1)

    # print(warped_sampling_grid_remake_pred.shape)
    # print(warped_sampling_grid_pred.shape)

    # final_transformation_matrix_exp = tf.matmul(expected_transformation_matrix, small_transform)[:3,:]
    # warped_sampling_grid_exp = tf.matmul(final_transformation_matrix_exp, homog_points_3d)
    #
    # points_2d_exp = tf.matmul(tf_K_mat, warped_sampling_grid_exp[:3,:])
    # Z_exp = points_2d_exp[2,:]
    # x_dash_exp = points_2d_exp[0,:]
    # y_dash_exp = points_2d_exp[1,:]
    # warped_sampling_grid_remake_exp = tf.stack([x_dash_exp, y_dash_exp, Z_exp], 1)
    #
    # print(warped_sampling_grid_remake_exp.shape)
    # print(warped_sampling_grid_remake_pred.shape)


    sparse_pred = sparsify_cloud(warped_sampling_grid_remake_pred)
    return sparse_pred
