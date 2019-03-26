'''
ECE421 A3 part 2

Code written by Andy Zhou

Mar 19, 2019
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import os
import math
import random
from tensorflow.python import debug as tf_debug

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Loading data
# data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

is_valid = False
# For Validation set
if is_valid:
    valid_batch = int(num_pts / 3.0)
    np.random.seed(0)
    rnd_idx = np.arange(num_pts)
    np.random.shuffle(rnd_idx)
    val_data = data[rnd_idx[:valid_batch]]
    data = data[rnd_idx[valid_batch:]]


# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)

    na = tf.reduce_sum(tf.square(X), 1)
    nb = tf.reduce_sum(tf.square(MU), 1)

    # na is N x K, nb is N x K
    # broadcast na along columns, broadcast nb along rows
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidean difference matrix
    D = na - 2 * tf.matmul(X, MU, False, True) + nb

    return D


def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1
    # log_pi: K X 1

    # Outputs:
    # log Gaussian PDF N X K

    # TODO
    D = X.shape[1]

    distance = distanceFunc(X, mu)

    invsigma = tf.math.reciprocal(sigma)

    invsigma = -1 / 2 * tf.square(invsigma)

    mat = tf.transpose(invsigma) * distance

    logsigma = -2 * tf.log(sigma)

    coeff = logsigma - tf.log(2 * tf.cast(np.pi, dtype=tf.float64))

    mat = mat + tf.transpose(coeff)

    return mat


def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # part 2.2
    # log P(Z = k| x) = log P(x|mu, sigma) + log_pi - logsumexp(z)

    term1 = tf.squeeze(log_pi) + log_PDF  # N x K

    log_post = term1 - tf.expand_dims(hlp.reduce_logsumexp(term1), 1)

    return log_post


def NLL_loss(log_gauss_PDF, log_pi):
    inner = tf.squeeze(log_pi) + logGaussPDF

    summed_over_clusters = hlp.reduce_logsumexp(inner)

    loss = -tf.reduce_sum(summed_over_clusters)

    return loss


num_updates = 600
lr = 0.01
K = 3

x = tf.placeholder(name='x', dtype=tf.float64, shape=(None, data.shape[1]))
mu = tf.get_variable(name='mean_vector', dtype=tf.float64, shape=(K, data.shape[1]),
                     initializer=tf.initializers.random_normal(seed=0))
phi = tf.get_variable(name='stdev_vector', dtype=tf.float64, shape=(K, 1),
                      initializer=tf.initializers.random_normal(seed=0))
psi = tf.get_variable(name='pi_vector', dtype=tf.float64, shape=(K, 1),
                      initializer=tf.initializers.random_normal(seed=0))

sigma = tf.exp(phi)

logGaussPDF = log_GaussPDF(x, mu, sigma)
logPi = hlp.logsoftmax(psi)

log_post = log_posterior(logGaussPDF, logPi)

NLLloss = NLL_loss(logGaussPDF, logPi)

optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(NLLloss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_loss_list = []

    for i in range(num_updates):
        sess.run(optimizer, feed_dict={x: data})

        train_loss = sess.run(NLLloss, feed_dict={x: data})

        train_loss_list.append(train_loss)

        print('The training loss is: {}'.format(train_loss))

    final_mu = sess.run(mu)
    final_sigma = sess.run(sigma)

    final_posterior = sess.run(log_post, feed_dict={x: data})

    print('the final mu is: ', final_mu)

    final_mu = sess.run(mu)
    loss_curve = plt.plot(train_loss_list)
    loss_curve = plt.xlabel('Number of updates')
    loss_curve = plt.ylabel('Loss')
    loss_curve = plt.title("K={}, Loss VS Number of updates".format(K))

    plt.show(loss_curve)

    data_cluster_mat = np.column_stack((data, np.ones((data.shape[0], 1))))

    for i, point in enumerate(data_cluster_mat):
        probabilities = final_posterior[i]
        point[2] = np.argmax(probabilities) + 1

    unique, counts = np.unique(data_cluster_mat[:, -1], return_counts=True)
    dict_counts = dict(zip(unique, counts))
    print(dict_counts)

    for cluster in range(1, K + 1):

        try:
            percentage = dict_counts[cluster] * 100 / data.shape[0]
            print('The percentage of points belonging to cluster {} is: {}% '.format(cluster, percentage))

        except KeyError:
            print('Cluster {} has no points belonging to it'.format(cluster))

    x_mu, y_mu = final_mu.T
    x, y, cluster_label = data_cluster_mat.T
    plt.scatter(x, y, c=cluster_label, label='data')
    plt.scatter(x_mu, y_mu, cmap='r', marker='X', label='centroids', c='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Result of running Gaussian Mixture Algorithm with K = {}'.format(K))
    plt.legend()
    plt.show()






