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
sys.path.insert(0, './dataset_files/')

import icl_lstm_loader as ldr

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
    # with tf.name_scope("Network"):
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
epoch = 0
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

                sequence_frames = (sequence_frames - 127.5)/127.5

                out = sess.run([loss, output_frames, train_step, merge], feed_dict={X:sequence_frames, depth_maps:sequence_depth, phase:True})
                print(out[0])

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

                    sequence_frames = (sequence_frames - 127.5)/127.5

                    out = sess.run([loss, output_frames, merge2], feed_dict={X:sequence_frames, depth_maps:sequence_depth, phase:False})
                    print(out[0])

                    writer.add_summary(out[2], total_iterations_validate)
                    total_iterations_validate+=1

        epoch+=1
        ldr.shuffle_sets()
