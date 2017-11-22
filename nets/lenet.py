# Deep Learning
# =============
# based on Udacity course UD730 https://classroom.udacity.com/courses/ud730, further modifications by vykozlov
#
# ---- 
# a-la LeNet neural network. Original LeNet5: http://yann.lecun.com/exdb/lenet/
# Here we take config similar to https://www.tensorflow.org/get_started/mnist/pros :
# [5,5,32]->max_pool->[5,5,64]->max_pool->FC->dropout->FC. ReLU as activation function.
# ----

import tensorflow as tf
import data_cfg as cfg
import utils.netutils as netutils

class LeNet(object):
	
	def __init__(self, x, keep_prob, num_classes):
		"""Create the graph of the AlexNet model.
		Args:
			x: Placeholder for the input tensor.
			keep_prob: Dropout probability.
			num_classes: Number of classes in the dataset.
		"""
		#Parse input arguments into class variables
		self.X = x
		self.NUM_CLASSES = num_classes
		self.KEEP_PROB = keep_prob
		
		# Call the create function to build the computational graph of LeNet
		self.LayerOut = self.build()

	def build(self):
		l1conv = netutils.cnn_layer(self.X, [5, 5, cfg.num_channels, 32], [32], 'conv1') # 20->50 # 32->64
		l1pool = netutils.max_pool_2x2(l1conv)
		lnconv = netutils.cnn_layer(l1pool, [5, 5, 32, 64], [64], 'conv2')
		lnpool = netutils.max_pool_2x2(lnconv)

		shape = lnpool.get_shape().as_list()
		print("shape: ", shape)
		lnreshaped = tf.reshape(lnpool, [-1, shape[1] * shape[2] * shape[3]]) #shape[0]
		lndim = shape[1] * shape[2] * shape[3]
		l3 = netutils.fc_layer(lnreshaped, lndim, 1024, 'layer3') #orig: image_size // 4 * image_size // 4 * depth
		l3drop = netutils.drop(l3,'drop_layer', self.KEEP_PROB)
		#l4 = nn_layer(l3drop, 512, 128, 'layer4', act=tf.nn.relu)
		#l5 = nn_layer(l4, 128, 256, 'layer5', act=tf.nn.relu)
		layerOut = netutils.fc_layer(l3drop, 1024, self.NUM_CLASSES, 'layer4', act=tf.identity)
		return layerOut
