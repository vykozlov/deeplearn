# Deep Learning
# =============
# based on Udacity course UD730 https://classroom.udacity.com/courses/ud730, further modifications by vykozlov
#  
# ---- 
# Helper Functions to build a neural network
# ----

import numpy as np
import tensorflow as tf
import data_cfg as cfg

# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
	"""Create a weight variable with appropriate initialization."""
	#initial = tf.truncated_normal(shape, stddev=0.1) # how to choose stddev?
	#return tf.Variable(initial,name="weight")
	# Xavier initializer
	#-initial = tf.get_variable("weight", shape=shape,
	#-                          initializer=tf.contrib.layers.xavier_initializer())
	initial = tf.contrib.layers.xavier_initializer()
	return tf.Variable(initial(shape=shape), name="weight")

def bias_variable(shape,const=0.1):
	"""Create a bias variable with appropriate initialization."""
	initial = tf.constant(const, shape=shape)
	#initial = tf.zeros(shape=shape)
	return tf.Variable(initial, name="bias")

def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		if not cfg.isFast:
			tf.summary.histogram('histogram', var)
        
def conv2d(dataset, stride, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(dataset, W, strides=stride, padding='SAME')

def max_pool_2x2(dataset):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(dataset, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  #1,2,2,1

#### --- Dropout --- ####
# Dropout - controls the complexity of the model, prevents co-adaptation of features.
# keep_prob - a place holder introduced below, the probability that each element is kept.
def drop(dataset,drop_layer, keep_prob=1.0):
	with tf.name_scope(drop_layer):
		tf.summary.scalar('dropout_keep_probability', keep_prob)
		dropped = tf.nn.dropout(dataset, keep_prob)
	return dropped  
    
def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
	"""Reusable code for making a simple neural net layer.

	It does a matrix multiply, bias add, and then uses relu to nonlinearize.
	It also sets up name scoping so that the resultant graph is easy to read,
	and adds a number of summary ops.
	"""
	# Adding a name scope ensures logical grouping of the layers in the graph.
	with tf.name_scope(layer_name):
		# This Variable will hold the state of the weights for the layer
		with tf.name_scope('weights'):
			weights = weight_variable([input_dim, output_dim])
			variable_summaries(weights)
		with tf.name_scope('biases'):
			biases = bias_variable([output_dim])
			variable_summaries(biases)
		with tf.name_scope('Wx_plus_b'):
			preactivate = tf.matmul(input_tensor, weights) + biases
			if not cfg.isFast:
				tf.summary.histogram('pre_activations', preactivate)
		activations = act(preactivate, name='activation')
		if not cfg.isFast:
			tf.summary.histogram('activations', activations)
		return activations    

def cnn_layer(indata, infilter, indepth, layer_name):
	"""Reusable code for making a convolution neural net layer.
	"""
	# Adding a name scope ensures logical grouping of the layers in the graph.
	with tf.name_scope(layer_name):
		# This Variable will hold the state of the weights for the layer
		with tf.name_scope('weights'):
			weights = weight_variable(infilter)
			variable_summaries(weights)
		with tf.name_scope('biases'):
			biases = bias_variable(indepth)
			variable_summaries(biases)
		with tf.name_scope('conv2d'):
			print(weights)
			conv = tf.nn.conv2d(indata, weights, [1, 1, 1, 1], padding='SAME') #1,2,2,1
			hidden = tf.nn.relu(conv + biases)
		outlayer = hidden
		return outlayer        

def check_accuracy(predictions, labels, method=1):
	if (method == 0):
		accu = 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]
	else:
		correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
		accu = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accu	
		
#In order to shuffle and sampling each mini-batch, the state whether a sample has been selected inside the current epoch should also be considered.
#Taken from: https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data
class Dataset:

	def __init__(self,data,labels):
		self._index_in_epoch = 0
		self._epochs_completed = 0
		self._data = data
		self._labels = labels
		self._num_examples = data.shape[0]
		pass

	@property
	def data(self):
		return self._data
		
	@property
	def labels(self):
		return self._labels
		
	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self,batch_size,shuffle = True):
		start = self._index_in_epoch		
		if start == 0 and self._epochs_completed == 0:
			idx = np.arange(0, self._num_examples)  # get all possible indexes
			np.random.shuffle(idx)  # shuffle indexe
			self._data = self.data[idx]  # get list of `num` random samples
			self._labels = self.labels[idx] # get list of `num` random labels

		# go to the next batch
		if start + batch_size > self._num_examples:
			self._epochs_completed += 1
			rest_num_examples = self._num_examples - start
			data_rest_part = self.data[start:self._num_examples]
			labels_rest_part = self.labels[start:self._num_examples]
			idx0 = np.arange(0, self._num_examples)  # get all possible indexes
			np.random.shuffle(idx0)  # shuffle indexes
			self._data = self.data[idx0]  # get list of `num` random samples
			self._labels = self.labels[idx0] # get list of `num` random labels

			start = 0
			self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
			end =  self._index_in_epoch  
			data_new_part =  self._data[start:end]
			labels_new_part =  self._labels[start:end]
			return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
		else:
			self._index_in_epoch += batch_size
			end = self._index_in_epoch
			return self._data[start:end], self._labels[start:end]