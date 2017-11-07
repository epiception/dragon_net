import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io, transform, img_as_ubyte, color
import sys
sys.dont_write_bytecode = True


batch_size = 1
IMG_HT = 480
IMG_WDT = 640

img_ht = tf.cast(tf.constant(IMG_HT), 'float32')
img_wdt = tf.cast(tf.constant(IMG_WDT), 'float32')

# Camera Parameters specific to ICL-Dataset

fx = 481.20
fy = -480.00
cx = 319.50
cy = 239.50

fx_scaled = 2*(fx)/640.0 #focal length x scaled for -1 to 1 range
fy_scaled = 2*(fy)/480.0 #focal length y scaled for -1 to 1 range
cx_scaled = -1 + 2*(cx - 1.0)/640.0 # optical center x scaled for -1 to 1 range
cy_scaled = -1 + 2*(cy - 1.0)/480.0 # optical center y scaled for -1 to 1 range

K_mat_scaled = np.array([[fx_scaled,  0.0, cx_scaled],
                         [0.0, fy_scaled,  cy_scaled],
                         [0.0, 0.0, 1.0]])
tf_K_mat = tf.constant(K_mat_scaled, dtype=tf.float32)

tf_fx_scaled = tf.constant(fx_scaled)
tf_fy_scaled = tf.constant(fy_scaled)
tf_cx_scaled = tf.constant(cx_scaled)
tf_cy_scaled = tf.constant(cy_scaled)

def _3D_transformer(input_map, depth_map, t_mat):

    cloud_height = tf.shape(input_map)[0]
    cloud_width = tf.shape(input_map)[1]

    batch_grids, transformed_depth_map = _3D_meshgrid_batchwise(IMG_HT, IMG_WDT, depth_map, batch_size, t_mat)

    x_all = batch_grids[0,:,:]
    y_all = batch_grids[1,:,:]

    return _bilinear_sampling(input_map, x_all, y_all)
    
def get_pixel_value(img ,x, y):

    indices = tf.stack([y, x], 2)
    return tf.gather_nd(img, indices)

def _3D_meshgrid_batchwise(height, width, depth_img, num_batch, transformation_matrix):
    """
    Creates 3d sampling meshgrid

    """

    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)

    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [1,-1])
    y_t_flat = tf.reshape(y_t, [1,-1])

    # reshape to (x_t, y_t , 1)
    ones = tf.ones_like(x_t_flat)

    ZZ = tf.reshape(depth_img, [-1])/tf.constant(5000.0) #5000 because constant factor for TUM RGBD Dataset

    sampling_grid_2d = tf.concat([x_t_flat, y_t_flat, ones], 0)

    projection_grid_3d = tf.matmul(tf.matrix_inverse(tf_K_mat), sampling_grid_2d*ZZ)

    homog_points_3d = tf.concat([projection_grid_3d, ones], 0)

    warped_sampling_grid = tf.matmul(transformation_matrix, homog_points_3d)

    uu_x = warped_sampling_grid[0]*tf_fx_scaled/warped_sampling_grid[2] + tf_cx_scaled
    vv_y = warped_sampling_grid[1]*tf_fy_scaled/warped_sampling_grid[2] + tf_cy_scaled

    uu_x = tf.expand_dims(uu_x, 0)
    vv_y = tf.expand_dims(vv_y, 0)

    reprojected_grid = tf.concat([uu_x, vv_y], 0)
    reprojected_grid = tf.reshape(reprojected_grid, (2, IMG_HT, IMG_WDT))
    transformed_depth = tf.reshape(warped_sampling_grid[2], (IMG_HT, IMG_WDT, 1))

    return reprojected_grid, transformed_depth


def _bilinear_sampling(img, x, y):
    """
    Sampling from input image and performing bilinear interpolation
    """

    max_y = tf.constant(IMG_HT - 1, dtype=tf.int32)
    max_x = tf.constant(IMG_WDT - 1, dtype=tf.int32)

    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W/H/D]
    x = 0.5 * ((x + 1.0) * tf.cast(IMG_WDT - 1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(IMG_HT - 1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

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

    # # add dimension for addition
    wa = tf.expand_dims(wa, axis=2)
    wb = tf.expand_dims(wb, axis=2)
    wc = tf.expand_dims(wc, axis=2)
    wd = tf.expand_dims(wd, axis=2)

    # # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    # Reduce output noise by culling values near 0
    neg_ones = tf.ones_like(out)*tf.constant(-1.0)
    upper_limit = tf.ones_like(out)*tf.constant(0.00001)
    lower_limit = tf.ones_like(out)*tf.constant(-0.00001)

    loc = tf.where(tf.logical_and(tf.greater_equal(out,lower_limit), tf.less_equal(out,upper_limit)), neg_ones, out)

    return loc
