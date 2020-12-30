from matplotlib.pyplot import imread
from tflearn.data_utils import *
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from numpy import array

pix_val = 255.0

channel = 3


model_path = "C:/Users/Alisa/Documents/bachelors/brand_logo_detection/logo_dataset.pickle"

image_file = "C:/Users/Alisa/Documents/bachelors/brand_logo_detection/unnamed.jpg"

im = Image.open(image_file)
width, height = im.size

#channel = im.shape

image_data = (imread(image_file).astype(float) - pix_val / 2) / pix_val

if image_data.shape != (height, width, channel):
    raise Exception('Unexpected image shape: %s' % str(image_data.shape))

model = pickle.load(open(model_path, 'rb'))
print(model)

an_image = Image.open(image_file)
image_sequence = an_image. getdata()
image_array = np.array(image_sequence)

print(image_array)
print(image_data)


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
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path="model/logo-classifier.tfl.ckpt")

model.load('logo-classifier.tfl')

def res_image(f, image_shape=[32,32], grayscale=False, normalize=True):
    img = load_image(f)
    width, height = img.size
    if width != image_shape[0] or height != image_shape[1]:
        img = resize_image(img, image_shape[0], image_shape[1])
    if grayscale:
        img = convert_color(img, 'L')
    elif img.mode == 'L':
        img = convert_color(img, 'RGB')

    img = pil_to_nparray(img)
    if normalize: # << this here is what you need
        img /= 255.
    img = array(img).reshape(1, image_shape[0], image_shape[1], 3)
    return img


img = res_image(image_file, [32,32], grayscale=False, normalize=True)

pred = model.predict(img)
print(" %s" % pred[0])


