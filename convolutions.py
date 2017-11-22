# coding: utf-8

# Deep Learning
# =============
# based on Udacity course UD730 https://classroom.udacity.com/courses/ud730, further modifications by vykozlov
# 
# ---- 
# Main python script to call
# ---- 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

import argparse
import sys
import os
import time

import data_cfg as cfg
import utils.ioutils as dlutils
import utils.netutils as netutils

FLAGS = None

# ---
# Let's build a small network with convolutional layers. 
# This program uses the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset to be used with python experiments. This dataset is designed to look like the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, while looking a little more like real data: it's a harder task, and the data is a lot less 'clean' than MNIST.
# ---


def main(_):
	
	# First download data, extract, and create pickle files per image, if do not exist.
	train_filename = dlutils.maybe_download(cfg.data_root, 'notMNIST_large.tar.gz', 247336696)
	test_filename = dlutils.maybe_download(cfg.data_root, 'notMNIST_small.tar.gz', 8458043)

	train_folders = dlutils.maybe_extract(cfg.data_root, train_filename)
	test_folders = dlutils.maybe_extract(cfg.data_root, test_filename)
	
	train_datasets = dlutils.maybe_pickle(train_folders, 45000)
	test_datasets = dlutils.maybe_pickle(test_folders, 1800)

	train_size = 200000
	valid_size = 10000
	test_size = 10000
	
	cfg.valid_dataset, cfg.valid_labels, cfg.train_dataset, cfg.train_labels = dlutils.merge_datasets(train_datasets, train_size, valid_size)
	_, _, cfg.test_dataset, cfg.test_labels = dlutils.merge_datasets(test_datasets, test_size)
	
	print('Training:', cfg.train_dataset.shape, cfg.train_labels.shape)
	print('Validation:', cfg.valid_dataset.shape, cfg.valid_labels.shape)
	print('Testing:', cfg.test_dataset.shape, cfg.test_labels.shape)
	
	cfg.train_dataset, cfg.train_labels = dlutils.randomize(cfg.train_dataset, cfg.train_labels)
	cfg.test_dataset, cfg.test_labels = dlutils.randomize(cfg.test_dataset, cfg.test_labels)
	cfg.valid_dataset, cfg.valid_labels = dlutils.randomize(cfg.valid_dataset, cfg.valid_labels)
	
	# Finally, let's save the data for later reuse, if don't yet exists
	force = cfg.force_rebuild
	if args["force_rebuild"] > 0:
		force = args["force_rebuild"]
		
	pickle_file = "notMNIST.pickle"
	if os.path.exists(pickle_file) and not force:
		# One may override by setting force=True.
		print('%s already present - Skipping saving pickle file' % pickle_file)
	else:
		datasets_dict = {'x_train' : cfg.train_dataset, 'y_train': cfg.train_labels,
						'x_valid': cfg.valid_dataset, 'y_valid': cfg.valid_labels,
						'x_test': cfg.test_dataset, 'y_test': cfg.test_labels}
		dlutils.store_datasets(cfg.data_root, pickle_file, datasets_dict)

	# "clean" the dataset from overlapping data
	pickle_cleanfile = 'notMNIST_clean.pickle' #notMNIST_clean.pickle
	if os.path.exists(pickle_cleanfile) and not force:
		# You may override by setting force=True.
		print('%s already present - Skipping saving cleaned pickle file' % pickle_cleanfile)
	else:
		print('\nMethod 3: hashlib and numpy')
		t1 = time.time()
		test_dataset_clean,test_labels_clean = dlutils.faster_overlaps_hashlib_and_numpy(cfg.data_root, pickle_cleanfile)
		t2 = time.time()
		print("Time: %0.2fs" % (t2 - t1))
		print(cfg.line_sep)
		
		#create corresponding .tfrecord files per set. We can skip this, as the files are not used later on!
		dlutils.convert_to_tfrecord(pickle_cleanfile, "train_dataset", "train_labels", "train.tfrecord")
		dlutils.convert_to_tfrecord(pickle_cleanfile, "valid_dataset", "valid_labels", "valid.tfrecord")
		dlutils.convert_to_tfrecord(pickle_cleanfile, "test_dataset", "test_labels", "test.tfrecord")
		
	with tf.device('/cpu:0'):		
		train_dataset, train_labels = dlutils.read_from_pickle(pickle_cleanfile, "train_dataset", "train_labels")
		valid_dataset, valid_labels = dlutils.read_from_pickle(pickle_cleanfile, "valid_dataset", "valid_labels")
		test_dataset, test_labels = dlutils.read_from_pickle(pickle_cleanfile, "test_dataset", "test_labels")
		print(cfg.line_sep)		
		print('Training set', train_dataset.shape, train_labels.shape)
		print('Validation set', valid_dataset.shape, valid_labels.shape)
		print('Test set', test_dataset.shape, test_labels.shape)
		print(cfg.line_sep)
	
		train_dataset, train_labels = dlutils.reformat(train_dataset, train_labels)
		valid_dataset, valid_labels = dlutils.reformat(valid_dataset, valid_labels)
		test_dataset, test_labels = dlutils.reformat(test_dataset, test_labels)
		print('Training set', train_dataset.shape, train_labels.shape)
		print('Validation set', valid_dataset.shape, valid_labels.shape)
		print('Test set', test_dataset.shape, test_labels.shape)
		print(cfg.line_sep)		


	from nets.lenet import LeNet
		
	mygraph = tf.Graph()	
	with mygraph.as_default():
		# Input data. For the training data, we use a placeholder that will be fed
		# at run time with a training minibatch.
		# Input placeholders
		with tf.name_scope('input'):
			x_dataset = tf.placeholder(tf.float32, shape=(None, cfg.image_size, cfg.image_size, cfg.num_channels))  # batch_size, image_size*image_size
			y_labels = tf.placeholder(tf.float32, shape=(None, cfg.num_labels))     # batch_size, 10

		keep_prob = tf.placeholder(tf.float32)			
		model = LeNet(x_dataset,keep_prob,cfg.num_labels)
		logits_out = model.LayerOut

		#### --- L2 Regularization --- ####
		vars   = tf.trainable_variables()
		print(vars)
		lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars
						if 'bias' not in v.name ]) * 0.001 # 0.0005
    
		with tf.name_scope('loss'):
			# The raw formulation of loss/cross-entropy,
			#
			# tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)), reduction_indices=[1]))
			#
			# can be numerically unstable.
			#
			# So here we use tf.nn.softmax_cross_entropy_with_logits on the
			# raw outputs of the nn_layer above, and then average across the batch.
			diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_labels, logits=logits_out)
			with tf.name_scope('total'):
				loss = tf.reduce_mean(diff + lossL2)
		tf.summary.scalar('loss', loss)

		#### --- Learning rate decay, see below --- ####
		with tf.name_scope('train'):
			# Optimizer. loss = loss function above
			# global_step is inspired by "Problem 4" below
			global_step = tf.Variable(0)  # count the number of steps taken.
			learning_rate = tf.train.exponential_decay(0.35, global_step, 500, 0.96, staircase=True) #learning_rate, global_step, decay_steps, decay_rate/momentum
			optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step) #0.5
			#optimizer = tf.train.AdamOptimizer(0.5e-4).minimize(loss) 
			# AdamOptimizer record:
			# 1e-3: 94% vs. 96% of sgd+expdecay (20k steps)
			# 1e-2: 88% vs. 96% of sgd+expdecay (20k steps)
			# 1e-4: 95.8% vs. 96.2 of sgd+expdecay (20k steps)
			# 1e-5: 93.6% (20k steps)
			# 1e-5: 95.8% (80k steps)
			# 1e-5: 96.3% (200k steps)
			# 0.5e-4:
        
		# Prediction
		y_prediction = tf.nn.softmax(logits_out)
		with tf.name_scope('accuracy'):
			accuracy = netutils.check_accuracy(y_prediction,y_labels)
		tf.summary.scalar('accuracy', accuracy)

		# Merge all the summaries and write them out to cfg.log_dir
		summaries = tf.summary.merge_all()


	#if number of epochs is specified, force this number, otherwise -> early stopping
	max_epochs = -1
	if args["max_epochs"] > 0:
		max_epochs = args["max_epochs"]
	
	print(cfg.line_sep)	
	print("Max number of epochs: ", max_epochs)
	print(cfg.line_sep)	

	### Optimization ###
	config = tf.ConfigProto()
	jit_level = 0
	if FLAGS.xla:
		# Turns on XLA JIT compilation.
		jit_level = tf.OptimizerOptions.ON_1
		print(cfg.line_sep)
		print("executing with XLA: ", jit_level)
		print(cfg.line_sep)
	jit_level = tf.OptimizerOptions.ON_1
	config.graph_options.optimizer_options.global_jit_level = jit_level # effect?
	config.intra_op_parallelism_threads = 2  # effect?
	config.inter_op_parallelism_threads = 2  # effect?

	# Run TF seesion:
	session = tf.Session(config=config)
	with tf.Session(graph=mygraph) as session:   
		# Clean directory for TensorBoard graphs
		import shutil
		with tf.device('/cpu:0'):
			if tf.gfile.Exists(cfg.log_dir):
				tf.gfile.DeleteRecursively(cfg.log_dir)
				# 'hack' to clean TensorBoard from older runs
				shutil.rmtree('log_dir', ignore_errors=True)
			tf.gfile.MakeDirs(cfg.log_dir)
		
		
			train_writer = tf.summary.FileWriter(cfg.log_dir + '/train', session.graph)
			valid_writer = tf.summary.FileWriter(cfg.log_dir + '/valid')
			test_writer = tf.summary.FileWriter(cfg.log_dir + '/test')
			
			train = netutils.Dataset(train_dataset, train_labels)

		# let's measure timing
		t0 = time.time()
		tprev = t0
		current_epoch = 0
		pcount = 0
		patience = 5
		step = 0
		
		tf.global_variables_initializer().run()
		print("Initialized")

		#for step in range(num_steps):
		while cfg.early_stopping:   #implement 'early stopping' with patience parameter. ea
			batch_data, batch_labels = train.next_batch(cfg.batch_size)

			# Prepare a dictionary telling the session where to feed the minibatch.
			# The key of the dictionary is the placeholder node of the graph to be fed,
			# and the value is the numpy array to feed to it.
			feed_train = {x_dataset : batch_data, y_labels : batch_labels, keep_prob : 0.5}  #0.5
			_, tloss, summary, predictions, accu = session.run(
				[optimizer, loss, summaries, y_prediction, accuracy], feed_dict=feed_train)
			if step == 0:
				prevloss = tloss
			if current_epoch < train.epochs_completed and (current_epoch + 1) == train.epochs_completed:
				current_epoch = train.epochs_completed
				print("..going to new epoch: ", current_epoch)
				print("..steps completed: %d" % step)
				
				valid_loss, valid_summary, valid_accu = session.run(
					[loss, summaries, accuracy],feed_dict={x_dataset : valid_dataset, y_labels : valid_labels, keep_prob : 1})
				valid_writer.add_summary(valid_summary, step)
				
				#'early stopping', let a little variation
				if valid_loss/prevloss > 0.995 and valid_loss/prevloss < 1.05:
					pcount += 1
				prevloss = valid_loss
				
				tstep = time.time()
				print("..timing for one epoch = %0.2fs" % (tstep - tprev))
				tprev = tstep
				
				print("Minibatch loss, accuracy: %.3f, %.1f%%" % (tloss, 100.*netutils.check_accuracy(predictions, batch_labels).eval()) )
				print("Minibatch accuracy (method 3): %.1f%%" % (100.*accu))				
				print("Validation: loss, accuracy: %.3f, %.1f%%" % (valid_loss, 100.*valid_accu))
				
				if pcount >= patience and max_epochs < 0:
					cfg.early_stopping = False
				elif current_epoch == max_epochs:
					cfg.early_stopping = False
					
				
			if (step % 10 == 0):
				train_writer.add_summary(summary, step)
					
			step += 1
				  
		print(cfg.line_sep)
		print("steps completed: %d" % step)
		print(cfg.line_sep)
		
		if (cfg.debug):
			vars   = tf.trainable_variables()
			for v in vars:
				print(v.name, ": ", v.eval())
				
		test_loss, test_summary, test_accu = session.run(
			[loss, summaries, accuracy],feed_dict={x_dataset : test_dataset, y_labels : test_labels, keep_prob : 1})
		print("\nTest: loss, accuracy (method 3): %.3f, %.1f%%\n" % (test_loss, 100.*test_accu))
		test_writer.add_summary(test_summary, step)
		if (cfg.debug):
			for v in vars:
				print(v.name, ": ", v.eval())
        
		with tf.device('/cpu:0'):
			train_writer.close()
			valid_writer.close()
			test_writer.close()
    
		tend = time.time()
		print("Total execution time = %0.2fs" % (tend - t0))
    
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--max-epochs", type=int, default=-1,
		help="(optional) number of epochs to train")
	parser.add_argument("--force-rebuild", type=bool, default=cfg.force_rebuild,
		help="(optional) force summary pickle files rebuild")
	parser.add_argument('--xla', type=bool, default=True, help='Turn xla via JIT On. OFF = --xla=\'\'')
	args = vars(parser.parse_args())
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

