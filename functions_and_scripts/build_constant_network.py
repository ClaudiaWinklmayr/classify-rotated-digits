import numpy as np
import tensorflow as tf

""" 
This function takes lists of weight and bias vectors and builds a
4-layer MLP with a softmax classifier. 
Call via: x, y, weights, bias, activations, measures = build_network(W, B)  
"""


def build_network(W, B): 
    x = tf.placeholder(tf.float32, shape=[784, None]) #inputs 
    y = tf.placeholder(tf.float32, shape=[10, None]) #outputs
    
    weights = {
    'W_h1': tf.Variable(W[0], dtype=tf.float32),
    'W_h2': tf.Variable(W[1], dtype=tf.float32),
    'W_h3': tf.Variable(W[2], dtype=tf.float32),
    'W_h4': tf.Variable(W[3], dtype=tf.float32), 
    'W_out': tf.Variable(W[4], dtype=tf.float32)
}
    biases = {
    'b_1': tf.Variable(B[0], dtype=tf.float32),
    'b_2': tf.Variable(B[1], dtype=tf.float32),
    'b_3': tf.Variable(B[2], dtype=tf.float32),
    'b_4': tf.Variable(B[3], dtype=tf.float32),
    'b_out': tf.Variable(B[4], dtype=tf.float32)
}

    h_1 = tf.nn.sigmoid(tf.matmul(weights['W_h1'], x) + biases['b_1'])
    h_2 = tf.nn.sigmoid(tf.matmul(weights['W_h2'], h_1) + biases['b_2'])
    h_3 = tf.nn.sigmoid(tf.matmul(weights['W_h3'], h_2) + biases['b_3'])
    h_4 = tf.nn.sigmoid(tf.matmul(weights['W_h4'], h_3) + biases['b_4'])
    
    # for some reason the tf implementation of softmax gave strange result. 
    # this is a handwritten version that works. h_out is only an intermediate 
    # step and should not be used for analysis
    h_out = tf.exp(tf.matmul(weights['W_out'], h_4) + biases['b_out']) 
    y_ = tf.divide(h_out, tf.reduce_sum(h_out))
    
    correct_prediction = tf.equal(tf.argmax(y,0), tf.argmax(y_,0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    activations = {'h_1': h_1, 'h_2': h_2, 'h_3': h_3, 'h_4': h_4, 'h_out': h_out, 'y_':y_}
    
    measures = {'correct_prediction' : correct_prediction, 'accuracy': accuracy}
    
    return x, y, weights, biases, activations, measures


