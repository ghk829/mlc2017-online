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

import ops as op

class alexnetModel(BaseModel):

  def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):

    with tf.name_scope('conv1layer'):
        conv1 = op.conv(model_input, 7, 96, 3)
        conv1 = op.lrn(conv1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

    # conv layer 2
    with tf.name_scope('conv2layer'):
        conv2 = op.conv(conv1, 5, 256, 1, 1.0)
        conv2 = op.lrn(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

    # conv layer 3
    with tf.name_scope('conv3layer'):
        conv3 = op.conv(conv2, 3, 384, 1)

    # conv layer 4
    with tf.name_scope('conv4layer'):
        conv4 = op.conv(conv3, 3, 384, 1, 1.0)

    # conv layer 5
    with tf.name_scope('conv5layer'):
        conv5 = op.conv(conv4, 3, 256, 1, 1.0)
        conv5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # fc layer 1
    with tf.name_scope('fc1layer'):
        fc1 = op.fc(conv5, 4096, 1.0)
        fc1 = tf.nn.dropout(fc1, 0.5)

    # fc layer 2
    with tf.name_scope('fc2layer'):
        fc2 = op.fc(fc1, 4096, 1.0)
        fc2 = tf.nn.dropout(fc2, 0.5)

    # fc layer 3 - output
    with tf.name_scope('fc3layer'):
        net = op.fc(fc2, num_classes, 1.0, None)

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
