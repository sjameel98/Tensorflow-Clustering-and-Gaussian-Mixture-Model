import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp


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
    diff = x2 - 2 * X @ tf.transpose(MU) + y2
    return diff


def kmeans(epochs, clusters, step, data, val_data):
    D = data.shape[1]
    N = data.shape[0]
    K = clusters

    #Initializing Placeholders & Variables
    centers = tf.get_variable(name='MU', dtype=tf.float64, shape=(K, D),
                              initializer=tf.initializers.random_normal(seed=1))
    inputs = tf.placeholder(name='inputs', dtype=tf.float64, shape=(None, D))

    training_losses = []
    valid_losses = []
    loss = tf.reduce_sum(tf.reduce_min(distanceFunc(inputs, centers), axis=1))
    optimizer = tf.train.AdamOptimizer(learning_rate=step, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            #Why do we feed in the same data twice?

            training_loss = sess.run(loss, feed_dict={inputs: data})
            valid_loss = sess.run(loss, feed_dict={inputs: val_data})
            sess.run(optimizer, feed_dict={inputs: data})
            training_losses.append(training_loss)
            valid_losses.append(valid_loss)
            print('Training loss = {} | Validation loss = {}'.format(training_loss, valid_loss))
        centroids = sess.run(centers)

        plt.plot(training_losses)
        plt.xlabel("Number Of Iterations")
        plt.ylabel("Loss")
        plt.title("Training Loss, K={}".format(K))
        plt.savefig("Loss{}Clusters".format(K))
        plt.show()

        #print(type(centroids))
        #print(centroids.shape)
        #print(data.shape)

        dist = distanceFunc(data, centroids)
        cluster_class = tf.argmin(dist, axis=1)
        cluster_class = cluster_class.eval()

        clusterID, count = np.unique(cluster_class, return_counts=True)
        counts = dict(zip(clusterID, count))

        print(clusterID,count)

        for i in range(K):
            p = counts[i]/N * 100
            print("Percentage of points in cluster {}: {}%".format(i, p))




        #print(cluster_class.shape)
        #print(cluster_class)

        datanp = data

        x = datanp[:, 0]
        y = datanp[:, 1]
        centroids_x = centroids[:, 0]
        centroids_y = centroids[:, 1]


        plt.scatter(x, y, c=cluster_class, label='Data Points', s=25, alpha=0.8, cmap = 'Dark2')

        mark = ['o', '+', 'h', 'd', '*', 'P']
        for i in range(centroids_x.shape[0]):
            plt.scatter(centroids_x[i], centroids_y[i], marker=mark[i], label='Cluster {} Mean'.format(i), s=100, c='k')



        #plt.scatter(centroids_x, centroids_y, marker='x', label='Cluster Means', c='k', s=100)
        plt.legend()
        plt.title("K-Means_{}Clusters".format(K))
        plt.savefig("K-Means_{}Clusters".format(K))
        plt.show()




if __name__ == "__main__":

    # Loading data
    data = np.load('data2D.npy')
    #data = np.load('data100D.npy')
    [num_pts, dim] = np.shape(data)

    # For Validation set
    is_valid = False
    if is_valid:
      valid_batch = int(num_pts / 3.0)
      np.random.seed(45689)
      rnd_idx = np.arange(num_pts)
      np.random.shuffle(rnd_idx)
      val_data = data[rnd_idx[:valid_batch]]
      data = data[rnd_idx[valid_batch:]]

    epochs = 600
    lr = 0.01
    K = 2




    kmeans(epochs = epochs, clusters = K, step = lr, data = data, val_data = data)