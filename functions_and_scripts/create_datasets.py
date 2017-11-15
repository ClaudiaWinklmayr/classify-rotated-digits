import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import pickle
from scipy.misc import imrotate

''' 
This script creates different versions of the MNIST test set (everything rotated 
by a certain angle or mixed rotations) and saves the resulting datasets as well as additional 
information (labels, indices of single digits, rotation angles) to a pickled dictionary 
''' 

FILENAME = 'all_mnist_variants.p'

#----------------------------------------------------------------------------------#
# Load MNIST test set and save to dictionary
#----------------------------------------------------------------------------------#
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

DATA_SETS = {}
DATA_SETS['mnist'] = mnist.test.images

DATA_POINTS = mnist.test.images.shape[0]
DATA_SHAPE = mnist.test.images.shape

#----------------------------------------------------------------------------------#
# Create test sets with single rotation angle
#----------------------------------------------------------------------------------#
mnist_30 = np.zeros(DATA_SHAPE)
mnist_60 = np.zeros(DATA_SHAPE)
mnist_90 = np.zeros(DATA_SHAPE)

mnist_neg30 = np.zeros(DATA_SHAPE)
mnist_neg60 = np.zeros(DATA_SHAPE)
mnist_neg90 = np.zeros(DATA_SHAPE)

for i in range(DATA_POINTS):
    im = np.reshape(mnist.test.images[i], (28,28))
    
    im30 = imrotate(im, 30, interp='bilinear')
    im60 = imrotate(im, 60, interp='bilinear')
    im90 = imrotate(im, 90, interp='bilinear')

    im_neg30 = imrotate(im, -30, interp='bilinear')
    im_neg60 = imrotate(im, -60, interp='bilinear')
    im_neg90 = imrotate(im, -90, interp='bilinear')
    
    mnist_30[i] = np.reshape(im30, (784,))
    mnist_60[i] = np.reshape(im60, (784,))
    mnist_90[i] = np.reshape(im90, (784,))

    mnist_neg30[i] = np.reshape(im_neg30, (784,))
    mnist_neg60[i] = np.reshape(im_neg60, (784,))
    mnist_neg90[i] = np.reshape(im_neg90, (784,))#
    
DATA_SETS['mnist_30'] = mnist_30
DATA_SETS['mnist_60'] = mnist_60
DATA_SETS['mnist_90'] = mnist_90

DATA_SETS['mnist_neg30'] = mnist_neg30
DATA_SETS['mnist_neg60'] = mnist_neg60
DATA_SETS['mnist_neg90'] = mnist_neg90

#----------------------------------------------------------------------------------#
# Create mixed set and save angles 
#----------------------------------------------------------------------------------#
angles = range(-90,90+1,15)
phis = np.zeros(DATA_POINTS)
mnist_mix = np.zeros(DATA_SHAPE)

for i in range(DATA_POINTS):
    im = np.reshape(mnist.test.images[i], (28,28))
    phi = random.sample(angles,1)[0]
    phis[i] = phi
    
    im_rot = imrotate(im, phi, interp='bilinear')
    mnist_mix[i] = np.reshape(im_rot, (784,))

DATA_SETS['mnist_mix'] = mnist_mix
DATA_SETS['mnist_mix_angles'] = phis


#----------------------------------------------------------------------------------#
# Save lables vector and dictionary with single digit indices
#----------------------------------------------------------------------------------#
single_numbers_test = {}
for i in range(10): 
    single_numbers_test['only_'+str(i)] = []
    
for idx in range(DATA_POINTS):
    d = np.flatnonzero(mnist.test.labels[idx])[0] 
    single_numbers_test['only_'+str(d)].append(idx)
    
    
DATA_SETS['digit_idx'] = single_numbers_test   
DATA_SETS['labels'] = mnist.test.labels




#----------------------------------------------------------------------------------#
# Save pickled data
#----------------------------------------------------------------------------------#
pickle.dump(DATA_SETS, open(FILENAME, 'wb'))



