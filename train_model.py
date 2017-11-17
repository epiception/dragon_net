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
import os

sys.dont_write_bytecode = True
sys.path.insert(0, './dataset_files/')

import icl_lstm_loader as ldr

file_path = os.path.dirname(os.path.abspath(__file__))

plt.ion()

batch_size = 20
sequence_size = 3
IMG_HT = 480
IMG_WDT = 640
NO_CHANNELS = 1
NO_CHANNEL_NET = 2

fc_output = 6
lstm_input = 1024
lstm_hidden = 1024

LR = 1e-6
BETA1 = 0.9

trained_weights = np.load("./pretrained_weights/model_weights.npy")[()]
trained_batchnorm = np.load("./pretrained_weights/model_batchnorm.npy")[()]

W_conv1 = init_weights_trained(trained_weights["weight_1"], 1, to_train = False)
W_conv2 = init_weights_trained(trained_weights["weight_2"], 2, to_train = False)
W_conv3 = init_weights_trained(trained_weights["weight_3"], 3, to_train = False)
W_conv4 = init_weights_trained(trained_weights["weight_4"], 4, to_train = False)
W_conv5 = init_weights_trained(trained_weights["weight_5"], 5, to_train = False)
W_conv6 = init_weights_trained(trained_weights["weight_6"], 6, to_train = False)
W_conv7 = init_weights_trained(trained_weights["weight_7"], 7, to_train = False)
W_conv8 = init_weights_trained(trained_weights["weight_8"], 8, to_train = False)
W_conv9 = init_weights_trained(trained_weights["weight_9"], 9, to_train = False)
W_conv10 = init_weights_trained(trained_weights["weight_10"], 10, to_train = False)
W_conv11 = init_weights_trained(trained_weights["weight_11"], 11, to_train = False)
W_fc1 = init_weights_trained(trained_weights["weight_12"], 12, to_train = False)
W_fc_lstm = weight_variable_fc([batch_size, 1024,6], 13)


weights = [W_conv1, W_conv2, W_conv3, W_conv4, W_conv5, W_conv6, W_conv7, W_conv8, W_conv9, W_conv10, W_conv11, W_fc1, W_fc_lstm]

weight_summaries = []

#for weight_index in range(len(weights)):
with tf.name_scope('W_fc_lstm'):
    current_sum = variable_summaries(weights[12])
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
    # with tf.name_scope("Network"):

    layer_1 = conv2d_batchnorm_load(input_x, weights[0], "conv_1", phase, [1,2,2,1], trained_batchnorm["BatchNorm/beta"], trained_batchnorm["BatchNorm/moving_mean"], trained_batchnorm["BatchNorm/moving_variance"], 0)

    layer_2 = conv2d_batchnorm_load(layer_1, weights[1], "conv_2", phase, [1,2,2,1], trained_batchnorm["BatchNorm_1/beta"], trained_batchnorm["BatchNorm_1/moving_mean"], trained_batchnorm["BatchNorm_1/moving_variance"], 1)

    layer_3 = conv2d_batchnorm_load(layer_2, weights[2], "conv_3", phase, [1,2,2,1], trained_batchnorm["BatchNorm_2/beta"], trained_batchnorm["BatchNorm_2/moving_mean"], trained_batchnorm["BatchNorm_2/moving_variance"], 2)

    layer_4 = conv2d_batchnorm_load(layer_3, weights[3], "conv_4", phase, [1,2,2,1], trained_batchnorm["BatchNorm_3/beta"], trained_batchnorm["BatchNorm_3/moving_mean"], trained_batchnorm["BatchNorm_3/moving_variance"], 3)

    layer_5 = conv2d_batchnorm_load(layer_4, weights[4], "conv_5", phase, [1,2,2,1], trained_batchnorm["BatchNorm_4/beta"], trained_batchnorm["BatchNorm_4/moving_mean"], trained_batchnorm["BatchNorm_4/moving_variance"], 4)

    layer_6 = conv2d_batchnorm_load(layer_5, weights[5], "conv_6", phase, [1,2,2,1], trained_batchnorm["BatchNorm_5/beta"], trained_batchnorm["BatchNorm_5/moving_mean"], trained_batchnorm["BatchNorm_5/moving_variance"], 5)

    layer_7 = conv2d_batchnorm_load(layer_6, weights[6], "conv_7", phase, [1,2,2,1], trained_batchnorm["BatchNorm_6/beta"], trained_batchnorm["BatchNorm_6/moving_mean"], trained_batchnorm["BatchNorm_6/moving_variance"], 6)

    layer_8 = conv2d_batchnorm_load(layer_7, weights[7], "conv_8", phase, [1,2,2,1], trained_batchnorm["BatchNorm_7/beta"], trained_batchnorm["BatchNorm_7/moving_mean"], trained_batchnorm["BatchNorm_7/moving_variance"], 7)

    layer_9 = conv2d_batchnorm_load(layer_8, weights[8], "conv_9", phase, [1,2,2,1], trained_batchnorm["BatchNorm_8/beta"], trained_batchnorm["BatchNorm_8/moving_mean"], trained_batchnorm["BatchNorm_8/moving_variance"], 8)

    layer_10 = conv2d_batchnorm_load(layer_9, weights[9], "conv_10", phase, [1,1,1,1], trained_batchnorm["BatchNorm_9/beta"], trained_batchnorm["BatchNorm_9/moving_mean"], trained_batchnorm["BatchNorm_9/moving_variance"], 9)

    layer_11 = conv2d_batchnorm_load(layer_10, weights[10], "conv_11", phase, [1,1,1,1], trained_batchnorm["BatchNorm_10/beta"], trained_batchnorm["BatchNorm_10/moving_mean"], trained_batchnorm["BatchNorm_10/moving_variance"], 10)

    layer_11_m = tf.reshape(layer_11, [1024,])

    return layer_11_m

X = tf.placeholder(tf.float32, [batch_size*sequence_size, IMG_HT, IMG_WDT, NO_CHANNEL_NET])
depth_maps = tf.placeholder(tf.float32, [batch_size*sequence_size, IMG_HT, IMG_WDT])
phase = tf.placeholder(tf.bool, [])

vec_out = tf.map_fn(lambda x: Net(X[x:x+1]), elems=tf.range(0, sequence_size*batch_size, 1), dtype=tf.float32)
lstm_out_xi = dynamic_RNN(vec_out, W_fc_lstm)

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
    # train_step = tf.train.AdamOptimizer(learning_rate=LR, beta1 = BETA1).minimize(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate= LR, beta1 = BETA1)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    new_grads, global_norm = tf.clip_by_global_norm(gradients, 5.0)
    train_step = optimizer.apply_gradients(zip(gradients, variables))

with tf.name_scope("Validation"):
    loss_validation = tf.nn.l2_loss(new_target_frames[:,32:-32,32:-32,:] - output_frames[:,32:-32,32:-32,:])

training_summary = tf.summary.scalar('Training_loss', loss)
validation_summary = tf.summary.scalar('Validation_loss', loss_validation)

saver = tf.train.Saver()

merge = tf.summary.merge([training_summary] + weight_summaries)
merge2 = tf.summary.merge([validation_summary])


##########################################################################################################################

total_train = 3780
total_validation = 720
partition_limit = 180
epoch = 70
n_epochs = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("./logs_3dstn_lstm/")

    total_iterations_train = 0
    total_iterations_validate = 0

    if(epoch == 0):
        writer.add_graph(sess.graph)

    if(epoch > 0):
        print("Restoring Checkoint")
        saver.restore(sess, "./Checkpoint_3d_lstm/model-%d.ckpt"%epoch)
        epoch+=1
        total_iterations_train = epoch*total_train/(batch_size*sequence_size)
        total_iterations_validate = epoch*total_validation/(batch_size*sequence_size)

    ##TRAINING##

    while(epoch < n_epochs):
        for part_idx in range(total_train/partition_limit):
            partition_frames, partition_depth = ldr.load_partition(part_idx, "Training")
            for batch_idx in range(0,partition_limit,batch_size*sequence_size):
                sequence_frames = partition_frames[batch_idx:batch_idx + batch_size*sequence_size]
                sequence_depth = partition_depth[batch_idx:batch_idx + batch_size*sequence_size]

                # sequence_frames = (sequence_frames - 127.5)/127.5
                sequence_frames = (sequence_frames - 0.5)/0.5

                out = sess.run([loss, output_frames, train_step, merge, lstm_out_xi, global_norm], feed_dict={X:sequence_frames, depth_maps:sequence_depth, phase:False})
                print(out[0], out[5])
                print(out[4][0])

                writer.add_summary(out[3], total_iterations_train)
                total_iterations_train+=1

        if (epoch%2 == 0):
            print("Saving after epoch %d"%epoch)
            saver.save(sess, "./Checkpoint_3d_lstm/model-%d.ckpt"%epoch )

        ##VALIDATION##

        for part_idx in range(total_validation/partition_limit):
            partition_frames, partition_depth = ldr.load_partition(part_idx, "Validation")
            for batch_idx in range(0,partition_limit,batch_size*sequence_size):
                sequence_frames = partition_frames[batch_idx:batch_idx + batch_size*sequence_size]
                sequence_depth = partition_depth[batch_idx:batch_idx + batch_size*sequence_size]

                # sequence_frames = (sequence_frames - 127.5)/127.5
                sequence_frames = (sequence_frames - 0.5)/0.5

                out = sess.run([loss, output_frames, merge2, lstm_out_xi], feed_dict={X:sequence_frames, depth_maps:sequence_depth, phase:False})
                print(out[0])
                print(out[3][0])

                writer.add_summary(out[2], total_iterations_validate)
                total_iterations_validate+=1

        epoch+=1
        ldr.shuffle_sets()
