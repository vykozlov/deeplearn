# deeplearn
Practicing Deep Learning based on the corresponding Udacity course UD730 (https://classroom.udacity.com/courses/ud730)

Code is structured as follows:

`convolutions.py` : main 'executable'

`data_cfg.py` : configuration file used across the project

`utils/ioutils.py` : set of functions to prepare data, shuffle, store it in .pickle files, load them later. One can also store data in .tfrecord file (to be skipped in future as since TF 1.3. DataSet API is introduced).

`utils/netutils.py` : set of help functions to construct a neural network

`nets/` : to put specific neural network structures here. Currently implemented: LeNet (see below)

**Currently implemented:**
* use Xavier initializer for weights (http://machinelearning.wustl.edu/mlpapers/papers/AISTATS2010_GlorotB10 )
* Optimization attempts with `tf.device('/cpu:0')` 
* use random batches but not repeat data within one epoch (based on the example taken from https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data )
* use either early stopping or predefined number of epochs
* implement class for LeNet neural network (Original LeNet5: http://yann.lecun.com/exdb/lenet/ . Here we take config similar to https://www.tensorflow.org/get_started/mnist/pros )
* use TensorBoard to monitor the network

In my own experiments a self-compiled TF 1.3.0 is used on my personal computer with Nvidia GTX1070.
Current LeNet implementation + batch_size=200 takes ca. 12s per epoch.


