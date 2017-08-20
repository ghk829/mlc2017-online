# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import models
import tensorflow as tf
import utils
from tensorflow import flags
FLAGS = flags.FLAGS
import tensorflow.contrib.slim as slim


class alexnetModel(BaseModel):

  def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):

    # conv layer 1
    conv1_weights = tf.Variable(tf.random_normal([7, 7, 1, 96], dtype=tf.float32, stddev=0.01))
    conv1_biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32))
    conv1 = tf.nn.conv2d(model_input, conv1_weights, [1, 3, 3, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, conv1_biases)
    conv1_relu = tf.nn.relu(conv1)
    conv1_norm = tf.nn.local_response_normalization(conv1_relu, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0)
    conv1_pool = tf.nn.max_pool(conv1_norm, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

    # conv layer 2
    conv2_weights = tf.Variable(tf.random_normal([5, 5, 96, 256], dtype=tf.float32, stddev=0.01))
    conv2_biases = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32))
    conv2 = tf.nn.conv2d(conv1_pool, conv2_weights, [1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, conv2_biases)
    conv2_relu = tf.nn.relu(conv2)
    conv2_norm = tf.nn.local_response_normalization(conv2_relu)
    conv2_pool = tf.nn.max_pool(conv2_norm, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

    # conv layer 3
    conv3_weights = tf.Variable(tf.random_normal([3, 3, 256, 384], dtype=tf.float32, stddev=0.01))
    conv3_biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32))
    conv3 = tf.nn.conv2d(conv2_pool, conv3_weights, [1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.bias_add(conv3, conv3_biases)
    conv3_relu = tf.nn.relu(conv3)

    # conv layer 4
    conv4_weights = tf.Variable(tf.random_normal([3, 3, 384, 384], dtype=tf.float32, stddev=0.01))
    conv4_biases = tf.Variable(tf.constant(1.0, shape=[384], dtype=tf.float32))
    conv4 = tf.nn.conv2d(conv3_relu, conv4_weights, [1, 1, 1, 1], padding='SAME')
    conv4 = tf.nn.bias_add(conv4, conv4_biases)
    conv4_relu = tf.nn.relu(conv4)

    # conv layer 5
    conv5_weights = tf.Variable(tf.random_normal([3, 3, 384, 256], dtype=tf.float32, stddev=0.01))
    conv5_biases = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32))
    conv5 = tf.nn.conv2d(conv4_relu, conv5_weights, [1, 1, 1, 1], padding='SAME')
    conv5 = tf.nn.bias_add(conv5, conv5_biases)
    conv5_relu = tf.nn.relu(conv5)
    conv5_pool = tf.nn.max_pool(conv5_relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # fc layer 1
    fc1_weights = tf.Variable(tf.random_normal([256 * 3 * 3, 4096], dtype=tf.float32, stddev=0.01))
    fc1_biases = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32))
    conv5_reshape = tf.reshape(conv5_pool, [-1, fc1_weights.get_shape().as_list()[0]])
    fc1 = tf.matmul(conv5_reshape, fc1_weights)
    fc1 = tf.nn.bias_add(fc1, fc1_biases)
    fc1_relu = tf.nn.relu(fc1)
    fc1_drop = tf.nn.dropout(fc1_relu, 0.5)

    # fc layer 2
    fc2_weights = tf.Variable(tf.random_normal([4096, 4096], dtype=tf.float32, stddev=0.01))
    fc2_biases = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32))
    fc2 = tf.matmul(fc1_drop, fc2_weights)
    fc2 = tf.nn.bias_add(fc2, fc2_biases)
    fc2_relu = tf.nn.relu(fc2)
    fc2_drop = tf.nn.dropout(fc2_relu, 0.5)

    # fc layer 3 - output
    fc3_weights = tf.Variable(tf.random_normal([4096, 10], dtype=tf.float32, stddev=0.01))
    fc3_biases = tf.Variable(tf.constant(1.0, shape=[10], dtype=tf.float32))
    fc3 = tf.matmul(fc2_drop, fc3_weights)
    net = tf.nn.bias_add(fc3, fc3_biases)

    output = slim.fully_connected(
        net, num_classes, activation_fn = tf.nn.softmax,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

  
class LogisticModel(BaseModel):

  def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):
    net = slim.flatten(model_input)
    output = slim.fully_connected(
        net, num_classes, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}
