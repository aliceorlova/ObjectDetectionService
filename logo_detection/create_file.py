from __future__ import division, print_function, absolute_import
import numpy as np
import scipy
from scipy import ndimage
import os
from six.moves import cPickle as pickle
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

width = 32
height = 32
channel = 3
pix_val = 255.0

dir = 'C:/Users/Alisa/Documents/bachelors/brand_logo_detection'
# Directory where processed images are stored
pp_dir = os.path.join(dir, '/flickr_logos_27_dataset/processedF')
# Name of the pickle file
pickle_file = 'logo_dataset.pickle'
# Number of images stored as training dataset in pickle file from the processed images
train_size = 70000
val_size = 5000
# Number of images stored as test dataset in pickle file from the processed images
test_size = 7000

# Creates array of dataset

def array(nb_rows, image_width, image_height, image_ch=1):
    if nb_rows:
        dataset = np.ndarray(                               #  stores its height, width and channel into an array
            (nb_rows, image_height, image_width, image_ch), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)        #  stores its labels
    else:
        dataset, labels = None, None
    return dataset, labels

# Merging pickle files of all the classes into one pickle file.
def combine(pickle_files, train_size, val_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = array(val_size, width,
                                              height, channel)
    train_dataset, train_labels = array(train_size, width,
                                              height, channel)
    vsize_per_class = val_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                logo_set = pickle.load(f)
                np.random.shuffle(logo_set)
                if valid_dataset is not None:
                    valid_logo = logo_set[:vsize_per_class, :, :, :]
                    valid_dataset[start_v:end_v, :, :, :] = valid_logo
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class
                train_logo = logo_set[vsize_per_class:end_l, :, :, :]
                train_dataset[start_t:end_t, :, :, :] = train_logo
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
    return valid_dataset, valid_labels, train_dataset, train_labels

def makepickle(train_dataset, train_labels, valid_dataset, valid_labels,
                test_dataset, test_labels):
    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)      # Saving data of the images into a pickle file
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise


# Opening the image
def load_logo(data_dir):
    image_files = os.listdir(data_dir)
    dataset = np.ndarray(
        shape=(len(image_files), height, width, channel),
        dtype=np.float32)
    print(data_dir)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(data_dir, image)
        try:
            image_data = (scipy.ndimage.imread(image_file).astype(float) -
                          pix_val / 2) / pix_val
            if image_data.shape != (height, width, channel):
                raise Exception('Unexpected image shape: %s' %
                                str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e,
                  '-it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    print('Full dataset tensor:', dataset.shape)       # Tell processed number of images for a particular class
    print('Mean:', np.mean(dataset))                   # Calculate mean over that entire class
    print('Standard deviation:', np.std(dataset))      # Calculate standard deviation over that entire class
    return dataset


def pickling(data_dirs, force=False):
    dataset_names = []
    for dir in data_dirs:
        set_filename = dir + '.pickle'
        dataset_names.append(set_filename)

        if os.path.exists(set_filename) and force:

            print('%s already present - Skipping pickling. ' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_logo(dir)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)
    return dataset_names

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

CLASS_NAME = [
    'Apple', 'BMW','Heineken','HP','Intel','Mini','Starbucks','Vodafone', 'Citroen', 'Ferrari'
]

dir2 = 'C:/Users/Alisa/Documents/bachelors/brand_logo_detection/flickr_logos_27_dataset/processedF'

dirs = [
        os.path.join(dir2, class_name, 'train').replace("\\","/")    # Look into all the train folder of the class
        for class_name in CLASS_NAME
    ]
test_dirs = [
        os.path.join(dir2, class_name, 'test').replace("\\","/")       # Look into all the test folder of the class
        for class_name in CLASS_NAME
    ]

print(dirs)
train_datasets = pickling(dirs)
test_datasets = pickling(test_dirs)

valid_dataset, valid_labels, train_dataset, train_labels = combine(train_datasets, train_size, val_size)# function called for merging
unknown1, unknown2, test_dataset, test_labels = combine(test_datasets, test_size)

train_dataset, train_labels = randomize(train_dataset, train_labels)   # function called for randomizing
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)

makepickle(train_dataset, train_labels, valid_dataset, valid_labels,test_dataset, test_labels)# function called for making a pickle file.
statinfo = os.stat(pickle_file)                         # Shows size of the file
print('Compressed pickle size:', statinfo.st_size)

def read_data():
    with open("logo_dataset.pickle", 'rb') as f:
        save = pickle.load(f)
        X = save['train_dataset']       # assign X as train dataset
        Y = save['train_labels']        # assign Y as train labels
        X_test = save['test_dataset']   # assign X_test as test dataset
        Y_test = save['test_labels']    #assign Y_test as test labels
        del save
    return [X, X_test], [Y, Y_test]

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, 32, 32,3)).astype(np.float32)    # Reformatting shape array to give a scalar value for dataset.
    labels = (np.arange(10) == labels[:, None]).astype(np.float32)
    return dataset, labels

dataset, labels = read_data()
X,Y = reformat(dataset[0], labels[0])
X_test, Y_test = reformat(dataset[1], labels[1])
print('Training set', X.shape, Y.shape)
print('Test set', X_test.shape, Y_test.shape)

# Shuffle the data
X, Y = shuffle(X, Y)    # Imported from TFLearn.

# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()


# 32x32 image is the input with 3 color channels (red, green and blue) with it's mean and standard deviation.
network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)


# Convolution 1 with 64 nodes with activation function as rectified linear unit:
network = conv_2d(network, 64, 3, activation='relu')

# Max pooling 1:
network = max_pool_2d(network, 2)

# Convolution 2 with 128 nodes with activation function as rectified linear unit:
network = conv_2d(network, 128, 3, activation='relu')

# Convolution 3 with 256 nodes with activation function as rectified linear unit:
network = conv_2d(network, 256, 3, activation='relu')

# Max pooling 2:
network = max_pool_2d(network, 2)

# Fully-connected 512 node neural network with activation function as rectified linear unit
network = fully_connected(network, 512 , activation='relu')

# Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Fully-connected neural network for outputs with activation function as softmax as we are dealing with multiclass classification.
network = fully_connected(network, 10, activation='softmax')

# To train the network we will use adaptive moment estimation (ADAM) and categorical_crossentropy
# to determine loss in learning process and optimization.
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

# Covering the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path="model\logo-classifier.tfl.ckpt")

# Training! n_epoch will tell how many iterations the network has to go through, here it is kept 15 training passes and monitor it as it goes.
model.fit(X,Y, n_epoch=10, shuffle=True, validation_set=(X_test, Y_test), show_metric=True, batch_size=128, snapshot_epoch=True,
          run_id='logo-classifier')

# Save model when training is complete to a file
model.save("logo-classifier.tfl")
print("Network trained and saved as logo-classifier.tfl!")


#Loading the trained dataset file 'logo-classifier.tfl.ckpt-4923'
model.load("model\logo-classifier.tfl.ckpt-4923")


#Evaluate the model
score=model.evaluate(X_test, Y_test)
print(score)

