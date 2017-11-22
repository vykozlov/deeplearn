# Deep Learning
# =============
# based on Udacity course UD730 https://classroom.udacity.com/courses/ud730, further modifications by vykozlov
#
# ---- 
# Configuration file
# Set here major parameters for the project
# ---- 
data_root = "/opt/wspace-valentin/dlearn"

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

batch_size = 200
early_stopping = True
	
log_dir = data_root+"/graphs"
line_sep = "-----------------------------"
isFast = True
debug = False
force_rebuild = False

train_dataset = None
train_labels  = None
valid_dataset = None
valid_labels  = None
test_dataset  = None
test_labels   = None
