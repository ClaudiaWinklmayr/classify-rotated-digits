import numpy as np
import pickle
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

""" 
SUMMARY: this script loads the weights of trained  caffe-AEs and uses them to initialize 
a tensorflow MLP. The softmax layer is trained, using the MNIST dataset. 
The resulting weights are saved
"""

# where the pretrained network weights are located
WEIGHTS_PATH = '/mnt/raid/ni/dnn/invariance/Claudia/Network_defs/weights_AER_AEUR/'
SAVE_PATH = WEIGHTS_PATH
WEIGHTS_NAMES = ['AER', 'AEUR'] 

# parameters for training the softmax layer
BATCH_SIZE = 128
TRAINING_ITER = 1000
LEARNING_RATE = 0.005

#load MNIST data for training
mnist = input_data.read_data_sets('/tmp/MNIST_data', one_hot=True)

for w in range(len(WEIGHTS_NAMES)): # iterate over all network types

    network_info = {}
    
    # load params and restructure bias for use with tesorflow
    params = pickle.load(open(WEIGHTS_PATH + WEIGHTS_NAMES[w] + '_params.p', 'rb'), encoding='latin1')
    W = params['weights']
    B = params['biases']
    for i in range(len(W)):
        B[i] = np.reshape(B[i], (len(B[i]), 1))

    # -------------------------------------------------------------------#
    #                 Retrain the Softmax layer                       #
    # -------------------------------------------------------------------#

    x = tf.placeholder(tf.float32, shape=[784, None]) #inputs 
    y = tf.placeholder(tf.float32, shape=[10, None]) #outputs

    W_h1 = tf.constant(W[0], dtype=tf.float32)
    b_1 = tf.constant(B[0], dtype=tf.float32)
    h1 = tf.nn.sigmoid(tf.matmul(W_h1, x) + b_1)

    W_h2 = tf.constant(W[1], dtype=tf.float32)
    b_2 = tf.constant(B[1], dtype=tf.float32)
    h2 = tf.nn.sigmoid(tf.matmul(W_h2, h1) + b_2)

    W_h3 = tf.constant(W[2], dtype=tf.float32)
    b_3 = tf.constant(B[2], dtype=tf.float32)
    h3 = tf.nn.sigmoid(tf.matmul(W_h3, h2) + b_3)

    W_h4 = tf.constant(W[3], dtype=tf.float32)
    b_4 = tf.constant(B[3], dtype=tf.float32)
    h4 = tf.nn.sigmoid(tf.matmul(W_h4, h3) + b_4)

    W_out = tf.Variable(W[4], dtype=tf.float32)
    b_out = tf.Variable(B[4], dtype=tf.float32)
    
 # the tensorflow softmax function gives strange outputs (i.e. only ones)
 # however in previous tests the result seemed fine. Still here we use the 
 # 'handwritten' version, just to be sure

 # y_ = tf.nn.softmax(tf.matmul(W_out, h4) + b_out)
    y_2 = tf.exp(tf.matmul(W_out, h4) + b_out)
    y_ = tf.divide(y_2, tf.reduce_sum(y_2))

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y,0), tf.argmax(y_,0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # here the training occurs, ce and acc are used to save values of cross entropy and acuracy for further analysis    
    ce = -99*np.ones((TRAINING_ITER,1))
    acc = -99*np.ones((TRAINING_ITER, 1))
    final_acc = -99


    for i in range(TRAINING_ITER):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict={x: batch_xs.T, y: batch_ys.T})
        ce[i] = sess.run(cross_entropy, feed_dict={x: batch_xs.T, y: batch_ys.T})
        acc[i] = sess.run(accuracy, feed_dict={x: batch_xs.T, y: batch_ys.T})

    final_acc = sess.run(accuracy, feed_dict={x: mnist.test.images.T, y: mnist.test.labels.T})

    weights_incl_softmax = []
    bias_incl_softmax = []

    weights_incl_softmax.append(sess.run(W_h1))
    bias_incl_softmax.append(sess.run(b_1))

    weights_incl_softmax.append(sess.run(W_h2))
    bias_incl_softmax.append(sess.run(b_2))

    weights_incl_softmax.append(sess.run(W_h3))
    bias_incl_softmax.append(sess.run(b_3))

    weights_incl_softmax.append(sess.run(W_h4))
    bias_incl_softmax.append(sess.run(b_4))

    weights_incl_softmax.append(sess.run(W_out))
    bias_incl_softmax.append(sess.run(b_out))

    network_info['CE_retrainingSoftmax'] = ce
    network_info['Acc_retrainingSoftmax'] = acc
    network_info['Accuracy_afterSoftmax'] = final_acc
    network_info['weights_incl_softmax'] = weights_incl_softmax
    network_info['bias_incl_softmax'] = bias_incl_softmax

    pickle.dump(network_info, open(SAVE_PATH + WEIGHTS_NAMES[w] + '_retrained_softmax_params.p' , 'wb'))

