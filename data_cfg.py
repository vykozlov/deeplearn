# Configuration file
# Set here major parameters for the project
#data_root = "/home/valentin/workspace/Udacity/dlearn"
data_root = "/opt/wspace-valentin/dlearn"

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

batch_size = 125
num_steps = 5001
	
log_dir = data_root+"/graphs"
isFast = True
debug = False
force_rebuild = False

train_dataset = None
train_labels  = None
valid_dataset = None
valid_labels  = None
test_dataset  = None
test_labels   = None
