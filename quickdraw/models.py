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
  
  

class YIGModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, model_input, num_classes = 10, l2_penalty = 1e-8, **unused_params):
  	with tf.variable_scope('Net') as sc:
  		IMAGE_SIZE = 50
  		
  		input = tf.image.resize_image_with_crop_or_pad(model_input, IMAGE_SIZE, IMAGE_SIZE)
  		input = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input)
  		input = tf.map_fn(lambda img: tf.image.per_image_standardization(img), input)
  		
  		net = slim.conv2d(input, 8, [3, 3], stride=1, activation_fn = tf.nn.relu, padding='SAME', scope='conv1')
	  	net = slim.max_pool2d(net, [2,2], stride=1, padding='SAME', scope='pool1')
	  	net = slim.conv2d(net, 4, [3, 3], stride=1, activation_fn = tf.nn.relu, padding='SAME', scope='conv2')
	  	net = slim.max_pool2d(net, [2,2], stride=1, padding='SAME', scope='pool2')
	  	net = slim.flatten(net)
	  	
	  	net = slim.dropout(net,0.5)
	  	
	  	output = slim.fully_connected(net, num_classes, activation_fn=tf.nn.softmax,
	  	weights_regularizer=slim.l2_regularizer(l2_penalty))
	  	
	  	return {"predictions": output}
	  	
  	raise NotImplementedError()
