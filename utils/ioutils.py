
# coding: utf-8


# Deep Learning
# =============
# 
# Assignment 1
# ------------
# 
# The objective of this assignment is to learn about simple data curation practices, and familiarize you with some of the data we'll be reusing later.
# 
# This notebook uses the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset to be used with python experiments. This dataset is designed to look like the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, while looking a little more like real data: it's a harder task, and the data is a lot less 'clean' than MNIST.


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

import data_cfg as cfg

# First, we'll download the dataset to our local machine. The data consists of characters rendered in a variety of fonts on a 28x28 image. The labels are limited to 'A' through 'J' (10 classes). The training set has about 500k and the testset 19000 labeled examples. Given these sizes, it should be possible to train models quickly on any machine.
url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(path, filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  dest_filename = os.path.join(path, filename)
  if force or not os.path.exists(dest_filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(dest_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', dest_filename)
  else:
    raise Exception(
      'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
  return dest_filename

# Extract the dataset from the compressed .tar.gz file.
# This should give you a set of directories, labeled A through J.

num_classes = 10
np.random.seed(133)

def maybe_extract(path, filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(path)
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  

# Now let's load the data in a more manageable format. Since, depending on your computer setup you might not be able to fit it all in memory, we'll load each class into a separate dataset, store them on disk and curate them independently. Later we'll merge them into a single dataset of manageable size.
# We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road. 
# A few images might not be readable, we'll just skip them.

#image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
	"""Load the data for a single letter label."""
	image_files = os.listdir(folder)
	dataset = np.ndarray(shape=(len(image_files), cfg.image_size, cfg.image_size),
                         dtype=np.float32)
	print(folder)
	num_images = 0
	for image in image_files:
		image_file = os.path.join(folder, image)
		try:
			image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
			if image_data.shape != (image_size, image_size):
				raise Exception('Unexpected image shape: %s' % str(image_data.shape))
			dataset[num_images, :, :] = image_data
			num_images = num_images + 1
		except IOError as e:
			print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
	dataset = dataset[0:num_images, :, :]
	if num_images < min_num_images:
		raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
	print('Full dataset tensor:', dataset.shape)
	print('Mean:', np.mean(dataset))
	print('Standard deviation:', np.std(dataset))
	return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

# ---
# Problem 2
# ---------
# 
# Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. Hint: you can use matplotlib.pyplot.
# 
# ---

def verify_img(pickle_file):
    #pickle_file = train_datasets[1]  # index 0 should be all As, 1 = all Bs, etc.
    with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)  # unpickle
        sample_idx = np.random.randint(len(letter_set))  # pick a random image index
        sample_image = letter_set[sample_idx, :, :]  # extract a 2D slice
        plt.figure()
        plt.imshow(sample_image)  # display it



# Merge and prune the training data as needed. Depending on your computer setup, you might not be able to fit it all in memory, and you can tune `train_size` as needed. The labels will be stored into a separate array of integers 0 through 9.
# Also create a validation dataset for hyperparameter tuning.
def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, cfg.image_size)
  train_dataset, train_labels = make_arrays(train_size, cfg.image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            

# Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

#.train_dataset, train_labels = randomize(train_dataset, train_labels)
#.test_dataset, test_labels = randomize(test_dataset, test_labels)
#.valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

# ---
# Problem 4
# ---------
# Convince yourself that the data is still good after shuffling!
# ---

def verify_img2(dataset, labels, sample_idx):
    sample_image = dataset[sample_idx, :, :]  # extract a 2D slice
    print("Idx: ", sample_idx, ", Label: ",labels[sample_idx])
    plt.figure()
    plt.imshow(sample_image)  # display it

#.idx = np.random.randint(len(train_dataset))  # pick a random image index
#.verify_img2(train_dataset,train_labels,idx)

# ---
# Problem 5
# ---------
# 
# By construction, this dataset might contain a lot of overlapping samples, including training data that's also contained in the validation and test set! Overlap between training and test can skew the results if you expect to use your model in an environment where there is never an overlap, but are actually ok if you expect to see training samples recur when you use it.
# Measure how much overlap there is between training, validation and test samples.
# 
# Optional questions:
# - What about near duplicates between datasets? (images that are almost identical)
# - Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
# ---

#.train_dataset.flags.writeable=False
#.test_dataset.flags.writeable=False
#.dup_table={}
#.verify_img2(train_dataset,train_labels,820)
#.verify_img2(train_dataset,train_labels,967)

#.train_counter = 0
#.for idx,img in enumerate(train_dataset):
#.    h = hash(bytes(img.data))
#.    if h in dup_table and (train_dataset[dup_table[h]].data == img.data):
#.       print('Duplicate image: %d matches %d' % (idx, dup_table[h]))
#.       train_counter += 1
#.    dup_table[h] = idx
#.test_counter = 0    
#.for idx,img in enumerate(test_dataset):
#.    h = hash(bytes(img.data))
#.    if h in dup_table and (train_dataset[dup_table[h]].data == img.data):
#.        print(test_counter,': Test image %d is in the training set (%d)' % (idx, dup_table[h]))
#.        test_counter += 1
#.        
#.print("Train duplicates: ", train_counter, "Test duplicates with train: ", test_counter)
        

# Finally, let's save the data for later reuse:
def store_datasets(path, pickle_file, dataset_dict):
    pickle_file = os.path.join(path, pickle_file)

    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': dataset_dict['x_train'],
            'train_labels': dataset_dict['y_train'],
            'valid_dataset': dataset_dict['x_valid'],
            'valid_labels': dataset_dict['y_valid'],
            'test_dataset': dataset_dict['x_test'],
            'test_labels': dataset_dict['y_test'],
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
        
    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)



# Different methods, comparison on what they give and how fast they are.
# Taken from https://discussions.udacity.com/t/assignment-1-problem-5/45657/21

def fast_overlaps_num_set_and_hash(images1, images2):
    images1.flags.writeable=False
    images2.flags.writeable=False
    hash1 = set([hash(bytes(image1.data)) for image1 in images1])
    hash2 = set([hash(bytes(image2.data)) for image2 in images2])
    all_overlaps = set.intersection(hash1, hash2)
    return len(all_overlaps)

def find_dups_and_overlaps(images1, images2):
    images1.flags.writeable=False
    images2.flags.writeable=False
    dup_table={}
    duplicates = []
    for idx,img in enumerate(images1):
        h = hash(bytes(img.data))
        if h in dup_table and (images1[dup_table[h]].data == img.data):
            duplicates.append((idx, dup_table[h]))
            #print 'Duplicate image: %d matches %d' % (idx, dup_table[h])
        dup_table[h] = idx
    overlaps = []
    for idx,img in enumerate(images2):
        h = hash(bytes(img.data))
        if h in dup_table and (images1[dup_table[h]].data == img.data):
            overlaps.append((dup_table[h], idx))
            #print 'Test image %d is in the training set' % idx
    return duplicates, overlaps

def num_overlaps_with_diff_labels(overlap_indices, labels1, labels2):
    count = 0
    for olap in overlap_indices:
        if labels1[olap[0]] != labels2[olap[1]]:
            count += 1
    return count

def faster_overlaps_hashlib_and_numpy(path, pickle_file):
    import hashlib
    train_hashes = [hashlib.sha1(x).digest() for x in cfg.train_dataset]
    valid_hashes = [hashlib.sha1(x).digest() for x in cfg.valid_dataset]
    test_hashes  = [hashlib.sha1(x).digest() for x in cfg.test_dataset]
	
	#.train_hashes = [hashlib.sha1(x).digest() for x in dataset_dict['x_train']]
    #.valid_hashes = [hashlib.sha1(x).digest() for x in dataset_dict['x_valid']]
    #.test_hashes  = [hashlib.sha1(x).digest() for x in dataset_dict['x_test']]
    
    valid_in_train = np.in1d(valid_hashes, train_hashes)
    test_in_train  = np.in1d(test_hashes,  train_hashes)
    test_in_valid  = np.in1d(test_hashes,  valid_hashes)
    
    valid_keep = ~valid_in_train
    test_keep  = ~(test_in_train | test_in_valid)
    
    valid_dataset_clean = cfg.valid_dataset[valid_keep]
    valid_labels_clean  = cfg.valid_labels [valid_keep]
    
    test_dataset_clean = cfg.test_dataset[test_keep]
    test_labels_clean  = cfg.test_labels [test_keep]
    
    datasets_dict = {'x_train' : cfg.train_dataset, 'y_train': cfg.train_labels,
                     'x_valid': valid_dataset_clean, 'y_valid': valid_labels_clean,
                     'x_test': test_dataset_clean, 'y_test': test_labels_clean}

    store_datasets(path, pickle_file, datasets_dict)
    
    print("valid -> train overlap: %d samples" % valid_in_train.sum())
    print("test  -> train overlap: %d samples" % test_in_train.sum())
    print("test  -> valid overlap: %d samples" % test_in_valid.sum())
    return test_dataset_clean,test_labels_clean


# ---
# Problem 6
# ---------
# 
# Let's get an idea of what an off-the-shelf classifier can give you on this data. It's always good to check that there is something to learn, and that it's a problem that is not so trivial that a canned solution solves it.
# 
# Train a simple model on this data using 50, 100, 1000 and 5000 training samples. Hint: you can use the LogisticRegression model from sklearn.linear_model.
# 
# Optional question: train an off-the-shelf model on all the data!
# 
# ---

# In[29]:


import pandas as pd

n_samples = 200000

def model_fit(mymodel,trainset,trainlabels,testset,testlabels,testsetclean,testlabelsclean):
	import time
	print("train dimensions: ", trainset.shape)
	print("train labels dimensions: ", trainlabels.shape)
	print("test dimensions: ", testset.shape)
	print("test labels dimensions: ", testlabels.shape)
	t1 = time.time()
	# TODO: train the classifier on the training data / labels:
	mymodel.fit(trainset,trainlabels)
	# TODO: score the classifier on the testing data / labels:
	score=mymodel.score(testset, testlabels)
	print("The Score: %0.3f %%" % round((score*100), 3))
	t2 = time.time()
	print("Time: %0.2fs" % (t2 - t1))
	# TODO: score the classifier on the testing data / labels (cleaned!):
	score=mymodel.score(testsetclean, testlabelsclean)
	print("The Score (cleaned set): %0.3f %%" % round((score*100), 3))
	t3 = time.time()
	print("Time: %0.2fs" % (t3 - t2))

#.# decision tree "as it is"
#.print("\nUsing Decision tree classifier...")
#.from sklearn import tree
#.treemodel = tree.DecisionTreeClassifier() #max_depth=9, criterion="entropy"
#.model_fit(treemodel,train_dataset2d[0:n_samples],train_labels[0:n_samples],test_dataset2d,test_labels,test_dataset2d_clean,test_labels_clean)

#.print("\nUsing Linear Regression classifier...")
#.from sklearn import linear_model
#.linearmodel = linear_model.LinearRegression()
#.model_fit(linearmodel,train_dataset2d[0:n_samples],train_labels[0:n_samples],test_dataset2d,test_labels,test_dataset2d_clean,test_labels_clean)


# Reformat into a shape that's more adapted to the models we're going to train:
# - data as a flat matrix,
# - labels as float 1-hot encodings.
def reformat(dataset, labels):
	dataset = dataset.reshape((-1, cfg.image_size, cfg.image_size, cfg.num_channels)).astype(np.float32)
	# Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
	labels = (np.arange(cfg.num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels
