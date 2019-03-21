import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
is_valid = True
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

epochs = 500
lr = 0.01
K = 3

# Distance function for K-means
def distanceFunc(X, MU):
  
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO
    # sum((x-y)^2) = sum(x^2 - 2xy + y^2)
  x2 = tf.reduce_sum(tf.square(X), 1)
  y2 = tf.reduce_sum(tf.square(MU), 1)
  x2 = tf.reshape(x2, [-1, 1])
  y2 = tf.reshape(y2, [1, -1])
  diff = x2 - 2 * X@tf.transpose(MU) + y2
  return diff

def kmeans(epochs, clusters, step, data):
  D = data.shape[1]
  N = data.shape[0]
  K = clusters
  
  centers = tf.get_variable(name = 'MU', dtype = tf.float64, shape = (K, D), initializer = tf.initializers.random_normal(seed = 1))
  inputs = tf.placeholder(name = 'inputs', dtype = tf.float64, shape = (None, D))
  
  training_losses = []
  valid_losses = []
  loss = tf.reduce_sum(tf.reduce_min(distanceFunc(inputs, centers), axis = 1))
  optimizer = tf.train.AdamOptimizer(learning_rate = step).minimize(loss)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
      training_loss = sess.run(loss, feed_dict = {inputs:data})
      valid_loss = sess.run(loss, feed_dict = {inputs:data})
      sess.run(optimizer, feed_dict = {inputs: data})
      training_losses.append(training_loss)
      valid_losses.append(valid_loss)
      print('Training loss = {} | Validation loss = {}'.format(training_loss, valid_loss))
    centroids = sess.run(centers)
    
    print(type(centroids))


kmeans(epochs = epochs, clusters = K, step = 0.01, data = data)