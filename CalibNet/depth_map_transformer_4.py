import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as smc
# import cv2
# import imageio as io

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

# def remove_zero_vecs(cloud):
#     intermediate_tensor = tf.reduce_sum(tf.abs(cloud), 1)
#     zero_vector = tf.zeros(shape=(1,1), dtype=tf.float32)
#     bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
#     omit_zeros_cloud = tf.boolean_mask(cloud, tf.transpose(bool_mask)[:,0])
#     return omit_zeros_cloud[:80000]

def _simple_transformer(depth_map, t_mat):
    batch_grids, transformed_depth_map = _3D_meshgrid_batchwise_diff(IMG_HT, IMG_WDT, depth_map, batch_size, t_mat)

    x_all = tf.reshape(batch_grids[:,0], (IMG_HT, IMG_WDT))
    y_all = tf.reshape(batch_grids[:,1], (IMG_HT, IMG_WDT))


    return _bilinear_sampling(transformed_depth_map, x_all, y_all)

    # return batch_grids, transformed_depth_map

    #return x_all, y_all, transformed_depth_vals

def _3D_meshgrid_batchwise_diff(height, width, depth_img, num_batch, transformation_matrix):
    """
    Creates 3d sampling meshgrid

    """

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

    final_transformation_matrix = tf.matmul(transformation_matrix, small_transform)[:3,:]
    warped_sampling_grid = tf.matmul(final_transformation_matrix, homog_points_3d)

    points_2d = tf.matmul(tf_K_mat, warped_sampling_grid[:3,:])

    Z = points_2d[2,:]
    print("Z.shape", Z.shape)

    # x = tf.expand_dims(tf.transpose(points_2d[0,:]/Z), 1)
    # y = tf.expand_dims(tf.transpose(points_2d[1,:]/Z), 1)

    x = tf.transpose(points_2d[0,:]/Z)
    y = tf.transpose(points_2d[1,:]/Z)

    mask_int = tf.cast(mask, 'int32')

    updated_indices = tf.expand_dims(tf.boolean_mask(mask_int*z_index, mask), 1)

    # reprojected_grid = tf.concat([x, y], 1)

    updated_Z = tf.scatter_nd(updated_indices, Z, tf.constant([width*height]))
    updated_x = tf.scatter_nd(updated_indices, x, tf.constant([width*height]))
    neg_ones = tf.ones_like(updated_x)*-1.0
    updated_x_fin = tf.where(tf.equal(updated_Z, zeros_target), neg_ones, updated_x)

    updated_y = tf.scatter_nd(updated_indices, y, tf.constant([width*height]))
    updated_y_fin = tf.where(tf.equal(updated_Z, zeros_target), neg_ones, updated_y)

    reprojected_grid = tf.stack([updated_x_fin, updated_y_fin], 1)
    # reprojected_grid = tf.reshape(reprojected_grid, (2, IMG_HT, IMG_WDT))

    transformed_depth = tf.reshape(updated_Z, (IMG_HT, IMG_WDT))

    return reprojected_grid, transformed_depth

def reverse_all(z):
    z = tf.cast(z, 'float32')
    w = tf.floor((tf.sqrt(8.*z + 1.) - 1.)/2.0)
    t = (w**2 + w)/2.0
    y = tf.clip_by_value(tf.expand_dims(z - t, 1), 0.0, IMG_HT - 1)
    x = tf.clip_by_value(tf.expand_dims(w - y[:,0], 1), 0.0, IMG_WDT - 1)

    # print(y.shape)
    # print(x.shape)
    #
    # print(tf.concat([y,x], 1).shape)

    return tf.concat([y,x], 1)

def get_pixel_value(img, x, y):

    """Cantor pairing for removing non-unique updates and indices. At the time of implementation, unfixed issue with scatter_nd causes problems with int32 update values. Till resolution, implemented on cpu """

    with tf.device('/cpu:0'):
        indices = tf.stack([y, x], 2)
        indices = tf.reshape(indices, (375*1242, 2))
        values = tf.reshape(img, [-1])

        Y = indices[:,0]
        X = indices[:,1]
        Z = (X + Y)*(X + Y + 1)/2 + Y

        filtered, idx = tf.unique(tf.squeeze(Z))
        updated_values  = tf.unsorted_segment_max(values, idx, tf.shape(filtered)[0])

        # updated_indices = tf.map_fn(fn=lambda i: reverse(i), elems=filtered, dtype=tf.float32)
        updated_indices = reverse_all(filtered)
        updated_indices = tf.cast(updated_indices, 'int32')
        resolved_map = tf.scatter_nd(updated_indices, updated_values, img.shape)

        return resolved_map

def _bilinear_sampling(img, x_func, y_func):
    """
    Sampling from input image and performing bilinear interpolation
    """

    max_y = tf.constant(IMG_HT - 1, dtype=tf.int32)
    max_x = tf.constant(IMG_WDT - 1, dtype=tf.int32)

    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W/H/D]
    x = 0.5 * ((x_func + 1.0) * tf.cast(IMG_WDT - 1, 'float32'))
    y = 0.5 * ((y_func + 1.0) * tf.cast(IMG_HT - 1, 'float32'))

    x = tf.clip_by_value(x, 0.0, tf.cast(max_x, 'float32'))
    y = tf.clip_by_value(y, 0.0, tf.cast(max_y, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, 0, max_x)
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)

    # find Ia, Ib, Ic, Id

    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    loc = wa*Ia + wb*Ib + wc*Ic + wd*Id

    # loc_shaped = tf.reshape(loc, [1, tf.shape(loc)[0], tf.shape(loc)[1], 1])
    # loc_pool = tf.nn.max_pool(loc, ksize=[1,2,3,1], strides=[1,1,1,1], padding="SAME")
    # return loc_pool[0,:,:,0]

    return loc
