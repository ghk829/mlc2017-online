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

def block1(net, scale=1.0, activation_fn=tf.nn.relu, padding='SAME',scope=None, reuse=None):  
	with tf.variable_scope(scope, 'Block', [net], reuse=reuse):
		with tf.variable_scope('Branch_0'):
		  tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
		  tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
									 padding=padding,
									 scope='Conv2d_1a_3x3')
		with tf.variable_scope('Branch_1'):
		  tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
		  tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
									  padding=padding,
									  scope='Conv2d_1a_3x3')
		with tf.variable_scope('Branch_2'):
		  tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
		  tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
									  scope='Conv2d_0b_3x3')
		  tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
									  padding=padding,
									  scope='Conv2d_1a_3x3')
		with tf.variable_scope('Branch_3'):
		  tower_pool = slim.max_pool2d(net, 3, stride=2,
									   padding=padding,
									   scope='MaxPool_1a_3x3')
		net = tf.concat(
			[tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)
		net = activation_fn(net)
	return net
def block2(net, scale=1.0, activation_fn=tf.nn.relu, padding='SAME',scope=None, reuse=None):  
	with tf.variable_scope(scope, 'Block', [net], reuse=reuse):
		with tf.variable_scope('Branch_0'):
		  tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
		  tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
									 padding=padding,
									 scope='Conv2d_1a_3x3')
		with tf.variable_scope('Branch_1'):
		  tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
		  tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
									  padding=padding,
									  scope='Conv2d_1a_3x3')
		with tf.variable_scope('Branch_3'):
		  tower_pool = slim.max_pool2d(net, 3, stride=2,
									   padding=padding,
									   scope='MaxPool_1a_3x3')
		net = tf.concat(
			[tower_conv_1, tower_conv1_1, tower_pool], 3)
		net = activation_fn(net)
	return net
def block3(net, scale=1.0, activation_fn=tf.nn.relu, padding='SAME',scope=None, reuse=None):  
	with tf.variable_scope(scope, 'Block', [net], reuse=reuse):
		with tf.variable_scope('Branch_0'):
		  tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
		  tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
									 padding=padding,
									 scope='Conv2d_1a_3x3')
		with tf.variable_scope('Branch_1'):
		  tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
		  tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
									  padding=padding,
									  scope='Conv2d_1a_3x3')
		net = tf.concat(
			[tower_conv_1, tower_conv1_1], 3)
		net = activation_fn(net)
	return net
class KHModel(models.BaseModel):

  def create_model(self, model_input, num_classes=2, l2_penalty=1e-8, **unused_params):
  	input = tf.map_fn(lambda img: tf.image.per_image_standardization(img), model_input,name='standardize')
  	net = slim.repeat(input, 9, block1)
	net=slim.flatten(net)
	net=slim.fully_connected(net,net.get_shape().as_list()[1]*2)
	output = slim.fully_connected(net,num_classes - 1, activation_fn=tf.nn.sigmoid,
	weights_regularizer=slim.l2_regularizer(l2_penalty))
	return {"predictions": output}
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
	
	
class JJModel(models.BaseModel):
	def create_model(self, model_input, num_classes=2, l2_penalty=1e-8, **unused_params):
	  	input = tf.map_fn(lambda img: tf.image.per_image_standardization(img), model_input,name='standardize')
	  	with tf.variable_scope('Net') as sc:
	  		net = slim.conv2d(input, 8, [3, 3], stride=1, activation_fn = tf.nn.relu,padding='SAME', scope='conv1')
		  	net = slim.max_pool2d(net, [2,2], stride=1,padding='SAME',scope='pool1')
		  	net = slim.conv2d(net, 16, [3, 3], stride=1, activation_fn = tf.nn.relu,padding='SAME', scope='conv2')
		  	net = slim.conv2d(net, 16, [3, 3], stride=1, activation_fn = tf.nn.relu, padding='SAME', scope='conv3')
		  	net = slim.max_pool2d(net, [2,2], stride=1, padding='SAME',scope='pool2')
		  	net = slim.flatten(net)
		  	net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu,scope='fc_1')
		  	net = slim.dropout(net,0.5)
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
