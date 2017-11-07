import numpy as np
import tensorflow as tf
import sys
sys.dont_write_bytecode = True

def exponential_map_single(vec):
    """
    Calulate exponential_map from single xi
    """
    with tf.name_scope("Exponential_map"):
        u = vec[:3]
        omega = vec[3:]

        theta = tf.sqrt(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2])

        omega_cross = tf.stack([0.0, -omega[2], omega[1], omega[2], 0.0, -omega[0], -omega[1], omega[0], 0.0])
        omega_cross = tf.reshape(omega_cross, [3,3])

        #Taylor's approximation for A,B and C not being used currently, approximations preferable for low values of theta

        # A = 1.0 - (tf.pow(theta,2)/factorial(3.0)) + (tf.pow(theta, 4)/factorial(5.0))
        A = tf.sin(theta)/theta

        # B = 1.0/factorial(2.0) - (tf.pow(theta,2)/factorial(4.0)) + (tf.pow(theta, 4)/factorial(6.0))
        B = (1.0 - tf.cos(theta))/(tf.pow(theta,2))

        # C = 1.0/factorial(3.0) - (tf.pow(theta,2)/factorial(5.0)) + (tf.pow(theta, 4)/factorial(7.0))
        C = (1.0 - A)/(tf.pow(theta,2))

        omega_cross_square = tf.matmul(omega_cross, omega_cross)

        R = tf.eye(3,3) + A*omega_cross + B*omega_cross_square
        V = tf.eye(3,3) + B*omega_cross + C*omega_cross_square
        Vu = tf.matmul(V,tf.expand_dims(u,1))

        T = tf.concat([R, Vu], 1)

        return T
