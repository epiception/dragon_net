import sys
import glob
import numpy as np
import tensorflow as tf
from math import factorial
import matplotlib.pyplot as plt
import scipy.misc as smc

from cnn_utils import *
import spatial_transformer_3d_modified_icl as st3
import Lie_functions as lie

sys.dont_write_bytecode = True

plt.ion()

batch_size = 2
sequence_size = 3
IMG_HT = 480
IMG_WDT = 640
NO_CHANNELS = 1
NO_CHANNEL_NET = 2

fc_output = 6
lstm_input = 1024
lstm_hidden = 1024

LR = 1e-4
BETA1 = 0.9

W_conv1 = weight_variable([3,3,NO_CHANNEL_NET,64], 1)
W_conv2 = weight_variable([3,3,64,64], 2)
W_conv3 = weight_variable([3,3,64,128], 3)
W_conv4 = weight_variable([3,3,128,256], 4)
W_conv5 = weight_variable([3,3,256,256], 5)
W_conv6 = weight_variable([3,3,256,512], 6)
W_conv7 = weight_variable([3,3,512,512], 7)
W_conv8 = weight_variable([3,3,512,1024], 8)
W_conv9 = weight_variable([3,3,1024,1024], 9)
W_conv10 = weight_variable([2,1,1024,1024], 10)
W_conv11 = weight_variable([1,1,1024,512], 11)
W_fc1 = weight_variable_fc([batch_size, 1024,6], 12)

weights = [W_conv1, W_conv2, W_conv3, W_conv4, W_conv5, W_conv6, W_conv7, W_conv8, W_conv9, W_conv10, W_conv11, W_fc1]

weight_summaries = []

for weight_index in range(len(weights)):
    with tf.name_scope('weight_%d'%weight_index):
        current_sum = variable_summaries(weights[weight_index])
        weight_summaries +=current_sum

def dynamic_RNN(x, weights):
    """
    Dynamic RNN Cell:

    Input: window of vectors
    Output: Scale consistent (time window) vectors

    """

    with tf.name_scope("Lstm"):
        x = tf.reshape(x, [batch_size, sequence_size, lstm_input])

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_hidden)

        rnn_outputs, state = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

        cell_op = rnn_outputs

        lstm_op =  tf.matmul(cell_op, weights)

        return lstm_op

def Net(input_x):
    with tf.name_scope("Network"):
        layer_1 = conv2d_batchnorm_init(input_x, weights[0], name="conv_1", phase=phase, stride=[1,2,2,1], padding="SAME")

        layer_2 = conv2d_batchnorm_init(layer_1, weights[1], name="conv_2", phase=phase, stride=[1,2,2,1], padding="SAME")

        layer_3 = conv2d_batchnorm_init(layer_2, weights[2], name="conv_3", phase=phase, stride=[1,2,2,1], padding="SAME")

        layer_4 = conv2d_batchnorm_init(layer_3, weights[3], name="conv_4", phase=phase, stride=[1,2,2,1], padding="SAME")

        layer_5 = conv2d_batchnorm_init(layer_4, weights[4], name="conv_5", phase=phase, stride=[1,2,2,1], padding="SAME")

        layer_6 = conv2d_batchnorm_init(layer_5, weights[5], name="conv_6", phase=phase, stride=[1,2,2,1], padding="SAME")

        layer_7 = conv2d_batchnorm_init(layer_6, weights[6], name="conv_7", phase=phase, stride=[1,2,2,1], padding="SAME")

        layer_8 = conv2d_batchnorm_init(layer_7, weights[7], name="conv_8", phase=phase, stride=[1,2,2,1], padding="SAME")

        layer_9 = conv2d_batchnorm_init(layer_8, weights[8], name="conv_9", phase=phase, stride=[1,2,2,1], padding="SAME")

        layer_10 = conv2d_batchnorm_init(layer_9, weights[9], name="conv_10", phase=phase, stride=[1,1,1,1], padding="SAME")

        layer_11 = conv2d_batchnorm_init(layer_10, weights[10], name="conv_11", phase=phase, stride=[1,1,1,1], padding="SAME")

        layer_11_m = tf.reshape(layer_11, [1024,])

        return layer_11_m

X = tf.placeholder(tf.float32, [batch_size*sequence_size, IMG_HT, IMG_WDT, NO_CHANNEL_NET])
depth_maps = tf.placeholder(tf.float32, [batch_size*sequence_size, IMG_HT, IMG_WDT])
phase = tf.placeholder(tf.bool, [])

vec_out = tf.map_fn(lambda x: Net(X[x:x+1]), elems=tf.range(0, sequence_size*batch_size, 1), dtype=tf.float32)
lstm_out_xi = dynamic_RNN(vec_out, W_fc1)

initial_frames = X[:,:,:,0:1]
target_frames = X[:,:,:,1:2]

with tf.name_scope("3D_Transformer"):
    output_frames = tf.map_fn(lambda y:
                        tf.map_fn(lambda x: st3._3D_transformer(initial_frames[y*sequence_size + x], depth_maps[y*sequence_size + x], t_mat = lie.exponential_map_single(lstm_out_xi[y,x])), elems=tf.range(0, sequence_size, 1), dtype = tf.float32),
                                elems=tf.range(0,batch_size,1), dtype=tf.float32)

output_frames = tf.reshape(output_frames, (sequence_size*batch_size, IMG_HT, IMG_WDT, 1))

new_depth_maps = tf.expand_dims(depth_maps, 3)
neg_ones = tf.ones_like(target_frames)*tf.constant(-1.0)
zeros = tf.zeros_like(target_frames)
new_target_frames = tf.where(tf.equal(new_depth_maps,zeros), neg_ones, target_frames)

with tf.name_scope("Training"):
    loss = tf.nn.l2_loss(new_target_frames[:,32:-32,32:-32,:] - output_frames[:,32:-32,32:-32,:])
    train_step = tf.train.AdamOptimizer(learning_rate=LR, beta1 = BETA1).minimize(loss)

##########################################################################################################################

frames_list = sorted(glob.glob("./frames/*.png"))
depth_list = sorted(glob.glob("./frames_depth/*.png"))

new_list = np.vstack((np.array(frames_list), np.roll(np.array(frames_list),-1)))


new_list = np.delete(new_list,[3,7] ,-1)

sequence_frames = np.zeros((batch_size*sequence_size, IMG_HT, IMG_WDT, NO_CHANNEL_NET), dtype=np.float32)
sequence_depth = np.zeros((batch_size*sequence_size, IMG_HT, IMG_WDT), dtype=np.float32)

seq_idx = 0
batch_idx = 0

for batch_idx in range(batch_size):
    for seq_idx in range(sequence_size):
        frame_1 = smc.imread(new_list[0, batch_idx*sequence_size + seq_idx], True)
        frame_2 = smc.imread(new_list[1, batch_idx*sequence_size + seq_idx], True)

        sequence_frames[batch_idx*sequence_size + seq_idx,:,:,0] = frame_1
        sequence_frames[batch_idx*sequence_size + seq_idx,:,:,1] = frame_2

        print(new_list[0, batch_idx*sequence_size + seq_idx], new_list[1, batch_idx*sequence_size + seq_idx])



for seq_idx in range(len(depth_list)):
    sequence_depth[seq_idx,:,:] = smc.imread(depth_list[seq_idx], True)
    print(depth_list[seq_idx])

sequence_frames = (sequence_frames - 127.5)/127.5


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out = sess.run([loss, output_frames, train_step], feed_dict={X:sequence_frames, depth_maps:sequence_depth, phase:True})

    index = 0
    for i in out[1]:
        plt.imshow(i[:,:,0], cmap = "gray")
        plt.pause(0.00001)
        plt.imshow((i[:,:,0] - sequence_frames[index,:,:,1])**2, cmap = "gray")
        plt.pause(0.00001)
        index+=1
