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

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils
from tensorflow import flags
FLAGS = flags.FLAGS
import tensorflow.contrib.slim as slim

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, num_classes=2, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    net = slim.flatten(model_input)
    output = slim.fully_connected(
        net, num_classes - 1, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MyModel(models.BaseModel):

  def create_model(self, model_input, num_classes=2, l2_penalty=1e-8, **unused_params):
  	net = slim.conv2d(model_input, 2, [2, 2], stride=1, padding='SAME', scope='conv1')
  	net = slim.max_pool2d(net, [2,2], padding='SAME',scope='pool1')
  	net = slim.conv2d(net, 3,[3,3],stride=2,padding='SAME',scope='conv2')
  	net = slim.max_pool2d(net, [2,2], padding='SAME',scope='pool2')
  	net = slim.flatten(net)
  	output = slim.fully_connected(net, num_classes - 1, activation_fn=tf.nn.sigmoid,
  	weights_regularizer=slim.l2_regularizer(l2_penalty))
  	return {"predictions": output}

    
class JJModel(models.BaseModel):

  def create_model(self, model_input, num_classes=2, l2_penalty=1e-8, **unused_params):
  	net = slim.conv2d(model_input, 2, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv1')
  	net = slim.conv2d(net, 2, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv2')
  	net = slim.max_pool2d(net, [2,2], stride=2, padding='SAME',scope='pool1')
  	net = slim.conv2d(net, 2, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv3')
  	net = slim.conv2d(net, 2, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv4')
  	net = slim.max_pool2d(net, [2,2], stride=2, padding='SAME',scope='pool2')
  	net = slim.conv2d(net, 2, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv5')
  	net = slim.conv2d(net, 2, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv6')
  	net = slim.conv2d(net, 2, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv7')
  	net = slim.conv2d(net, 2, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv8')
  	net = slim.max_pool2d(net, [2,2], stride=2, padding='SAME',scope='pool3')
  	net = slim.conv2d(net, 2, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv9')
  	net = slim.conv2d(net, 2, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv10')
  	net = slim.conv2d(net, 2, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv11')
  	net = slim.conv2d(net, 2, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv12')
  	net = slim.max_pool2d(net, [2,2], stride=2, padding='SAME',scope='pool4')
  	net = slim.conv2d(net, 2, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv13')
  	net = slim.conv2d(net, 2, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv14')
  	net = slim.conv2d(net, 2, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv15')
  	net = slim.conv2d(net, 2, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='conv16')
  	net = slim.max_pool2d(net, [2,2], stride=2, padding='SAME',scope='pool5')
  	net = slim.flatten(net)
  	output = slim.fully_connected(net, num_classes - 1, activation_fn=tf.nn.sigmoid,
  	weights_regularizer=slim.l2_regularizer(l2_penalty))
  	return {"predictions": output}
  	
	

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size=2,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}
    
    
class CNNModel(models.BaseModel):
  """CNN Model """
  def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):
    """Creates a CNN model.
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
   
    """ model_input = tf.reshape(model_input,[-1,32,32,1])
    W = tf.Variable(tf.random_normal([3, 3, 1, 32]))
    strides=1
    p_keep_conv=0.5
    p_keep_hidden=0.7"""
    w = tf.Variable(tf.random_normal([3, 3, 1, 32]))
    w2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))     
    w3 = tf.Variable(tf.random_normal([3, 3, 64, 128]))    
    w4 = tf.Variable(tf.random_normal([128 * 4 * 4, 625])) 
    w_o = tf.Variable(tf.random_normal([625, 10]))
    """net = slim.conv2d(model_input,32,[5,5],scope='convl')
    net = slim.flatten(net)"""
    l1a = tf.nn.relu(tf.nn.conv2d(model_input, w,                       
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],             
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)
    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)
    output = slim.fully_connected(
        net, num_classes - 1, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}
  