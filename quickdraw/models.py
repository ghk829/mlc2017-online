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

from libs.connections import conv2d, linear
from collections import namedtuple
from math import sqrt

"""Contains the base class for models."""
class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, unused_model_input, **unused_params):
    raise NotImplementedError()

class LogisticModel(BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      num_classes: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    net = slim.flatten(model_input)
    output = slim.fully_connected(
        net, num_classes, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}
  
  

class ResnetModel(BaseModel):
# %%
  def residual_network(self, model_input, n_outputs = 10,
						 activation=tf.nn.relu, l2_penalty = 1e-8 ):
	# %%
	LayerBlock = namedtuple(
		'LayerBlock', ['num_repeats', 'num_filters', 'bottleneck_size'])
	blocks = [LayerBlock(3, 128, 32),
			  LayerBlock(3, 256, 64),
			  LayerBlock(3, 512, 128),
			  LayerBlock(3, 1024, 256)]
	# %%
	input_shape = model_input.get_shape().as_list()
	if len(input_shape) == 2:
		ndim = int(sqrt(input_shape[1]))
		if ndim * ndim != input_shape[1]:
			raise ValueError('input_shape should be square')
		model_input = tf.reshape(model_input, [-1, ndim, ndim, 1])
	# %%
	# First convolution expands to 64 channels and downsamples
	net = conv2d(model_input, 64, k_h=7, k_w=7,
				 name='conv1',
				 activation=activation)
	# %%
	# Max pool and downsampling
	net = tf.nn.max_pool(
		net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
	# %%
	# Setup first chain of resnets
	net = conv2d(net, blocks[0].num_filters, k_h=1, k_w=1,
				 stride_h=1, stride_w=1, padding='VALID', name='conv2')
	# %%
	# Loop through all res blocks
	for block_i, block in enumerate(blocks):
		for repeat_i in range(block.num_repeats):

			name = 'block_%d/repeat_%d' % (block_i, repeat_i)
			conv = conv2d(net, block.bottleneck_size, k_h=1, k_w=1,
						  padding='VALID', stride_h=1, stride_w=1,
						  activation=activation,
						  name=name + '/conv_in')

			conv = conv2d(conv, block.bottleneck_size, k_h=3, k_w=3,
						  padding='SAME', stride_h=1, stride_w=1,
						  activation=activation,
						  name=name + '/conv_bottleneck')

			conv = conv2d(conv, block.num_filters, k_h=1, k_w=1,
						  padding='VALID', stride_h=1, stride_w=1,
						  activation=activation,
						  name=name + '/conv_out')

			net = conv + net
		try:
			# upscale to the next block size
			next_block = blocks[block_i + 1]
			net = conv2d(net, next_block.num_filters, k_h=1, k_w=1,
						 padding='SAME', stride_h=1, stride_w=1, bias=False,
						 name='block_%d/conv_upscale' % block_i)
		except IndexError:
			pass
	# %%
	net = tf.nn.avg_pool(net,
						 ksize=[1, net.get_shape().as_list()[1],
								net.get_shape().as_list()[2], 1],
						 strides=[1, 1, 1, 1], padding='VALID')
	net = tf.reshape(
		net,
		[-1, net.get_shape().as_list()[1] *
		 net.get_shape().as_list()[2] *
		 net.get_shape().as_list()[3]])
	
	output = slim.fully_connected(
		net, n_outputs, activation_fn = tf.nn.softmax,
		weights_regularizer=slim.l2_regularizer(l2_penalty))
	return {"predictions": output}
