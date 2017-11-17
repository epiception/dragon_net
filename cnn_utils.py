import numpy as np
import tensorflow as tf


def weight_variable(shape, layerno):
    '''
    Helper function to create a weight variable initialized with
    a normal distribution (truncated to two standard devs)
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''

    W = tf.get_variable("weight_%d"%layerno, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return W

def weight_variable_fc(shape, layerno):
    '''
    Helper function to create a weight variable initialized with
    a normal distribution (truncated to two standard devs)
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''

    W = 0.005*tf.get_variable("weight_%d"%layerno, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    # W = tf.get_variable("weight_%d"%layerno, shape=shape, initializer=tf.zeros_initializer())

    return W

def weight_variable_fc_small(shape, layerno):
    '''
    Helper function to create a weight variable initialized with
    a normal distribution (truncated to two standard devs)
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''

    #W = tf.get_variable("weight_%d"%layerno, shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.05))
    W = 0.0005*tf.get_variable("weight_%d"%layerno, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    #W = tf.get_variable("weight_%d"%layerno, shape=shape, initializer=tf.zeros_initializer)

    return W

def bias_variable(layerno ,shape):

    B = tf.Variable(tf.constant(0.0, shape= shape, dtype=tf.float32), name="bias_%d"%layerno)
    return B

def init_weights_trained(W, layerno, to_train):
    return tf.Variable(W, trainable=to_train, name='weight_%d'%layerno)

def init_bias_trained(B, layerno, to_train):
    return tf.Variable(B, trainable=to_train, name='bias_%d'%layerno)


def conv2d_batchnorm_load(x, W, name, phase, stride, beta_r, mean_r, variance_r, layerno):

    beta = tf.constant_initializer(beta_r)
    moving_mean = tf.constant_initializer(mean_r)
    moving_variance = tf.constant_initializer(variance_r)

    with tf.name_scope(name):
        mid1 =  tf.nn.conv2d(x, W, strides = stride, padding = "SAME", name="conv_%d"%layerno)
        mid2 = tf.nn.relu(mid1, 'relu')

        with tf.name_scope('batch_norm'):
            return tf.contrib.layers.batch_norm(mid2, param_initializers={'beta': beta, 'moving_mean': moving_mean,'moving_variance': moving_variance,}, is_training = phase, updates_collections = None)

def conv2d_batchnorm_init(x, W, name, phase, stride, padding):
    with tf.name_scope(name):
        mid1 =  tf.nn.conv2d(x, W, strides = stride, padding = padding)
        mid2 = tf.nn.relu(mid1, 'relu')
        with tf.name_scope('batch_norm'):
            return tf.contrib.layers.batch_norm(mid2, is_training = phase, updates_collections = None)
            #return tf.nn.relu(mid2, 'relu')

def conv2d_init(x, W, name, phase, stride, padding):
    with tf.name_scope(name):
        mid1 =  tf.nn.conv2d(x, W, strides = stride, padding = padding)
        mid2 = tf.nn.relu(mid1, 'relu')

        return mid2

def max_pool(x, name):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME", name=name)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    sum_mean = tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    sum_stddev = tf.summary.scalar('stddev', stddev)
    #tf.summary.scalar('max', tf.reduce_max(var))
    #tf.summary.scalar('min', tf.reduce_min(var))
    sum_hist = tf.summary.histogram('histogram', var)
    return [sum_mean, sum_hist, sum_stddev]
