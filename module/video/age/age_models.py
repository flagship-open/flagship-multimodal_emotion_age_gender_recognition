import tensorflow as tf
import numpy as np
import sys
sys.path.append('./age/new')
from .new.facenet import FaceNet

class ImageNetFaceNet(FaceNet):
    def normalize(self, x):
        res = (x - np.reshape([127.5, 127.5, 127.5], [1, 1, 1, 3])) / 127.5
        return res

class AgeModelMorph:
    def build(self):
        self.pretrained = ImageNetFaceNet()
        self.min_age = tf.constant(16, name='min_age')
        self.max_age = tf.constant(77, name='max_age')
        self.imgs = tf.placeholder(dtype=tf.float32, shape=self.pretrained.input_shape, name='input_images')
        self.ages = tf.placeholder(dtype=tf.int32, shape=[None], name='age_labels_in_num')
        self.bottleneck = self.pretrained.construct(self.imgs, False, 1.0)
        self.feat_norm = tf.nn.l2_normalize(self.bottleneck, axis=1)

        self.fc1 = self.fc_bn(self.feat_norm, 128, activation=tf.nn.relu, name='fc1',
                              kernel_regularizer=lambda w: tf.nn.l2_loss(w))
        self.fc2 = tf.layers.dense(self.fc1, 62, activation=None, name='fc2',
                                   kernel_regularizer=lambda w: tf.nn.l2_loss(w))
        self.softmax = tf.nn.softmax(self.fc2)
        self.predicted_class = tf.cast(tf.argmax(self.softmax, axis=1), tf.int32)
        self.predicted_age = self.min_age + self.predicted_class
        self.target_class = tf.cast(self.ages - self.min_age, tf.int32)
        self.mae = tf.reduce_mean(tf.cast(tf.abs(self.predicted_class - self.target_class), tf.float32))

    def get_test_output(self):
        # input must include 1 instance augmented with crops
        out_response = tf.reduce_mean(self.fc2, axis=0)

        predicted_class = tf.cast(tf.argmax(out_response, axis=0), tf.int32)
        predicted_age = self.min_age + predicted_class
        return predicted_age

    def fc_bn(self, input, dim, activation, name, kernel_regularizer=None):
        fc1 = tf.layers.dense(input, dim, name=name, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=kernel_regularizer)
        fc1_bn = tf.layers.batch_normalization(
                    inputs=fc1,
                    axis=-1,
                    momentum=0.9,
                    epsilon=0.001,
                    center=True,
                    scale=True,
                    training = False,
                    name=name+'_bn'
                )
        return activation(fc1_bn, name=name+'_a')