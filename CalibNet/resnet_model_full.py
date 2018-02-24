import numpy as np
import tensorflow as tf
import scipy.misc as smc
import matplotlib.pyplot as plt

import config_res as config
from cnn_utils_res import *
import resnet_rgb_model as model
import resnet_depth_model as model_depth
import depth_map_transformer_4 as st3
import depth_map_and_cloud_transformer_emd as ct3
import nw_loader_2_color as ldr
import model_utils


plt.ion()

IMG_HT = config.depth_img_params['IMG_HT']
IMG_WDT = config.depth_img_params['IMG_WDT']
batch_size = config.net_params['batch_size']
learning_rate = config.net_params['learning_rate']
n_epochs = config.net_params['epochs']

_BETA_CONST = config.net_params['beta_const']
_ALPHA_CONST = config.net_params['alpha_const']
current_epoch = config.net_params['current_epoch']

W_ext1 = weight_variable([3,3,768,384], "_8")
W_ext2 = weight_variable([3,3,384,384], "_9")
W_ext3 = weight_variable([1,2,384,384], "_10")

W_ext4_rot = weight_variable([1,1,384,384], "_11")
W_fc_rot = weight_variable_fc([3840,3], "_12")

W_ext4_tr = weight_variable([1,1,384,384], "_13")
W_fc_tr = weight_variable_fc([3840,3], "_14")

end_weights = [W_ext1, W_ext2, W_ext3, W_ext4_rot, W_fc_rot, W_ext4_tr, W_fc_tr]

weight_summaries = []

for weight_index in range(len(end_weights)):
    with tf.name_scope('weight_%d'%weight_index):
        weight_summaries += variable_summaries(end_weights[weight_index])

def End_Net(input_x, phase_depth, weights = end_weights):

    layer8 = conv2d_batchnorm_init(input_x, weights[0], name="conv_9", phase= phase_depth, stride=[1,2,2,1])
    layer9 = conv2d_batchnorm_init(layer8, weights[1], name="conv_10", phase= phase_depth, stride=[1,2,2,1])
    layer10 = conv2d_batchnorm_init(layer9, weights[2], name="conv_11", phase= phase_depth, stride=[1,1,1,1])

    layer11_rot = conv2d_batchnorm_init(layer10, weights[3], name="conv_12", phase= phase_depth, stride=[1,1,1,1])
    layer11_m_rot = tf.reshape(layer11_rot, [batch_size, 3840])
    layer11_drop_rot = tf.nn.dropout(layer11_m_rot, keep_prob)
    layer11_vec_rot = (tf.matmul(layer11_drop_rot, weights[4]))

    layer11_tr = conv2d_batchnorm_init(layer10, weights[5], name="conv_13", phase= phase_depth, stride=[1,1,1,1])
    layer11_m_tr = tf.reshape(layer11_tr, [batch_size, 3840])
    layer11_drop_tr = tf.nn.dropout(layer11_m_tr, keep_prob)
    layer11_vec_tr = (tf.matmul(layer11_drop_tr, weights[6]))

    output_vectors = tf.concat([layer11_vec_tr, layer11_vec_rot], 1)
    return output_vectors


def exponential_map_single(vec):

    "Decoupled for SO(3) and translation t"

    with tf.name_scope("Exponential_map"):
        # t = tf.expand_dims(vec[:3], 1)
        u = vec[:3]
        omega = vec[3:]

        theta = tf.sqrt(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2])

        omega_cross = tf.stack([0.0, -omega[2], omega[1], omega[2], 0.0, -omega[0], -omega[1], omega[0], 0.0])
        omega_cross = tf.reshape(omega_cross, [3,3])

        #Taylor's approximation for A,B and C not being used currently, approximations preferable for low values of theta

        # A = 1.0 - (tf.pow(theta,2)/factorial(3.0)) + (tf.pow(theta, 4)/factorial(5.0))
        # B = 1.0/factorial(2.0) - (tf.pow(theta,2)/factorial(4.0)) + (tf.pow(theta, 4)/factorial(6.0))
        # C = 1.0/factorial(3.0) - (tf.pow(theta,2)/factorial(5.0)) + (tf.pow(theta, 4)/factorial(7.0))

        A = tf.sin(theta)/theta

        B = (1.0 - tf.cos(theta))/(tf.pow(theta,2))

        C = (1.0 - A)/(tf.pow(theta,2))

        omega_cross_square = tf.matmul(omega_cross, omega_cross)

        R = tf.eye(3,3) + A*omega_cross + B*omega_cross_square

        V = tf.eye(3,3) + B*omega_cross + C*omega_cross_square
        Vu = tf.matmul(V,tf.expand_dims(u,1))

        T = tf.concat([R, Vu], 1)
        # T = tf.concat([R, t], 1)
        # T  = tf.concat([T, tf.constant(np.array([[0.0, 0.0, 0.0, 1.0]]), dtype = tf.float32)], 0)

        return T

X1 = tf.placeholder(tf.float32, shape = (batch_size, IMG_HT, IMG_WDT, 3))
X2 = tf.placeholder(tf.float32, shape = (batch_size, IMG_HT, IMG_WDT, 1))
depth_maps_target = tf.placeholder(tf.float32, shape = (batch_size, IMG_HT, IMG_WDT, 1))
expected_transforms = tf.placeholder(tf.float32, shape = (batch_size, 4, 4))

phase = tf.placeholder(tf.bool, [])
phase_rgb = tf.placeholder(tf.bool, [])

keep_prob = tf.placeholder(tf.float32)

X2_pooled = tf.nn.max_pool(X2, ksize=[1,5,5,1], strides=[1,1,1,1], padding="SAME")
depth_maps_target_pooled = tf.nn.max_pool(depth_maps_target, ksize=[1,5,5,1], strides=[1,1,1,1], padding="SAME")
pooled_input = X2_pooled


RGB_Net_obj = model.Resnet(X1, phase_rgb)
Depth_Net_obj = model_depth.Depthnet(X2_pooled, phase)

with tf.name_scope('ResNet'):
    output_rgb = RGB_Net_obj.Net()
with tf.variable_scope('DepthNet'):
    output_depth = Depth_Net_obj.Net()

layer_next = tf.concat([output_rgb, output_depth], 3)
output_vectors = End_Net(layer_next, phase)
predicted_transforms = tf.map_fn(lambda x:exponential_map_single(output_vectors[x]), elems=tf.range(0, batch_size, 1), dtype=tf.float32)

depth_maps_predicted = tf.map_fn(lambda x:st3._simple_transformer(X2_pooled[x,:,:,0]*40.0 + 40.0, predicted_transforms[x]), elems = tf.range(0, batch_size, 1), dtype = tf.float32)

depth_maps_expected = tf.map_fn(lambda x:st3._simple_transformer(X2_pooled[x,:,:,0]*40.0 + 40.0, expected_transforms[x]), elems = tf.range(0, batch_size, 1), dtype = tf.float32)

cloud_pred = tf.map_fn(lambda x:ct3._cloud_transformer(X2[x,:,:,0]*40.0 + 40.0, predicted_transforms[x]), elems = tf.range(0, batch_size, 1), dtype = tf.float32)

cloud_exp = tf.map_fn(lambda x:ct3._cloud_transformer(X2[x,:,:,0]*40.0 + 40.0, expected_transforms[x]), elems = tf.range(0, batch_size, 1), dtype = tf.float32)

photometric_loss = tf.nn.l2_loss(tf.subtract((depth_maps_expected[:,10:-10,10:-10] - 40.0)/40.0, (depth_maps_predicted[:,10:-10,10:-10] - 40.0)/40.0))

cloud_loss = model_utils.get_emd_loss(cloud_pred, cloud_exp)
# cloud_loss = tf.nn.l2_loss(cloud_pred - cloud_exp)

predicted_loss_train = _ALPHA_CONST*photometric_loss + _BETA_CONST*cloud_loss

tf.add_to_collection('losses1', predicted_loss_train)
loss1 = tf.add_n(tf.get_collection('losses1'))

train_step = tf.train.AdamOptimizer(learning_rate = config.net_params['learning_rate'],
                                    beta1 = config.net_params['beta1']).minimize(loss1)

predicted_loss_validation = tf.nn.l2_loss(tf.subtract((depth_maps_expected[:,10:-10,10:-10] - 40.0)/40.0, (depth_maps_predicted[:,10:-10,10:-10] - 40.0)/40.0))

cloud_loss_validation = model_utils.get_emd_loss(cloud_pred, cloud_exp)
# cloud_loss_validation = tf.nn.l2_loss(cloud_pred - cloud_exp)

training_summary_1 = tf.summary.scalar('cloud_loss', _BETA_CONST*cloud_loss)
training_summary_2 = tf.summary.scalar('photometric_loss', photometric_loss)
validation_summary_1 = tf.summary.scalar('Validation_loss', predicted_loss_validation)
validation_summary_2 = tf.summary.scalar('Validation_cloud_loss', cloud_loss_validation)

saver = tf.train.Saver()

merge_train = tf.summary.merge([training_summary_1] + [training_summary_2] + weight_summaries)
# merge_train = tf.summary.merge([training_summary_2] + weight_summaries)
merge_val = tf.summary.merge([validation_summary_1] + [validation_summary_2])

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("./logs_simple_transformer/")

    total_iterations_train = 0
    total_iterations_validate = 0

    # if(current_epoch == 0):
    #     writer.add_graph(sess.graph)

    if(current_epoch > 0):
        print("Restoring Checkpoint")

        checkpoints_file_name = "./Checkpoint_simple_transformer/model-%d"%current_epoch

        saver = tf.train.import_meta_graph(checkpoints_file_name + '.meta')
        saver.restore(sess, checkpoints_file_name)
        current_epoch+=1
        total_iterations_train = current_epoch*config.net_params['total_frames_train']/batch_size
        total_iterations_validate = current_epoch*config.net_params['total_frames_validation']/batch_size

    for epoch in range(current_epoch, n_epochs):
        total_partitions_train = config.net_params['total_frames_train']/config.net_params['partition_limit']
        total_partitions_validation = config.net_params['total_frames_validation']/config.net_params['partition_limit']
        ldr.shuffle()

        for part in range(total_partitions_train):
            source_container, target_container, source_img_container, target_img_container, transforms_container = ldr.train_load(part)

            for source_b, target_b, source_img_b, target_img_b, transforms_b in zip(source_container, target_container, source_img_container, target_img_container, transforms_container):

                outputs= sess.run([depth_maps_predicted, depth_maps_expected, predicted_loss_train, X2_pooled, train_step, merge_train, predicted_transforms, cloud_loss, photometric_loss, loss1], feed_dict={X1: source_img_b, X2: source_b, depth_maps_target: target_b, expected_transforms: transforms_b ,phase:True, keep_prob:0.5, phase_rgb: False})

                # outputs= sess.run([depth_maps_predicted, depth_maps_expected, predicted_loss_train, X2_pooled, train_step, merge_train, predicted_transforms, photometric_loss, loss1], feed_dict={X1: source_img_b, X2: source_b, depth_maps_target: target_b, expected_transforms: transforms_b ,phase:True, keep_prob:0.7})

                dmaps_pred = outputs[0]
                dmaps_exp = outputs[1]
                loss = outputs[2]
                source = outputs[3]

                if(total_iterations_train%10 == 0):
                    writer.add_summary(outputs[5], total_iterations_train/10)
                # total_iterations_train+=1

                print(outputs[8], _ALPHA_CONST*outputs[8], outputs[7], _BETA_CONST*outputs[7], outputs[9],total_iterations_train)

                random_disp = np.random.randint(batch_size)
                print(outputs[6][random_disp])
                print(transforms_b[random_disp])

                if(total_iterations_train%125 == 0):

                    # plt.imshow(np.vstack((source[random_disp,:,:,0]*40.0 + 40.0, dmaps_pred[random_disp], dmaps_exp[random_disp])))
                    smc.imsave("training_imgs_iterative/training_save_%d.png"%total_iterations_train, np.vstack((source[random_disp,:,:,0]*40.0 + 40.0, dmaps_pred[random_disp], dmaps_exp[random_disp])))

                # if(total_iterations_train%1000 == 0):
                #     np.savetxt("training_imgs_iterative/cloud_pred_%d.txt"%total_iterations_train, outputs[10].reshape(64*4096, 3), fmt='%.5f')
                #     np.savetxt("training_imgs_iterative/cloud_exp_%d.txt"%total_iterations_train, outputs[11].reshape(64*4096, 3), fmt='%.5f')

                total_iterations_train+=1

        if (epoch%2 == 0):
            print("Saving after epoch %d"%epoch)
            saver.save(sess, "./Checkpoint_simple_transformer/model-%d"%epoch )



        for part in range(total_partitions_validation):
            source_container, target_container, source_img_container, target_img_container, transforms_container = ldr.validation_load(part)

            for source_b, target_b, source_img_b, target_img_b, transforms_b in zip(source_container, target_container, source_img_container, target_img_container, transforms_container):

                outputs= sess.run([depth_maps_predicted, depth_maps_expected, predicted_loss_validation, X2_pooled, merge_val, cloud_loss_validation], feed_dict={X1: source_img_b, X2: source_b, depth_maps_target: target_b, expected_transforms: transforms_b ,phase:False, keep_prob:1.0, phase_rgb: False})

                dmaps_pred = outputs[0]
                dmaps_exp = outputs[1]
                loss = outputs[2]
                source = outputs[3]

                writer.add_summary(outputs[4], total_iterations_validate)
                total_iterations_validate+=1

                print(loss, total_iterations_validate, outputs[5])

                if(total_iterations_validate%25 == 0):

                    random_disp = np.random.randint(batch_size)
                    # plt.imshow(np.vstack((source[random_disp,:,:,0]*40.0 + 40.0, dmaps_pred[random_disp], dmaps_exp[random_disp])))
                    smc.imsave("validation_imgs_iterative/validation_save_%d.png"%total_iterations_validate, np.vstack((source[random_disp,:,:,0]*40.0 + 40.0, dmaps_pred[random_disp], dmaps_exp[random_disp])))
                    # plt.pause(0.001)


            # break
        # break
