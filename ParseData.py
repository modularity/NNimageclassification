import scipy.io


fileName = 'caltech101_silhouettes_28_split1.mat'
images = scipy.io.loadmat(fileName)

# Put them in python arrays
train_data = images['train_data']
train_labels = images['train_labels']
test_data = images['test_data']
test_labels = images['test_labels']
val_data = images['val_data']
val_labels = images['val_labels']

# Process them to put them into array
classnames = [];
for classname in images['classnames'][0]:
	classnames.append(classname[0].encode('ascii'))



