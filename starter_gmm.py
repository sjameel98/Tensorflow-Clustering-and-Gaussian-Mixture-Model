import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp


# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO
    x2 = tf.reduce_sum(tf.square(X), 1)
    y2 = tf.reduce_sum(tf.square(MU), 1)
    x2 = tf.reshape(x2, [-1, 1])
    y2 = tf.reshape(y2, [1, -1])
    diff = x2 - 2 * X @ tf.transpose(MU) + y2
    return diff


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

    invsigma = -1/2 * tf.square(invsigma)

    mat = tf.transpose(invsigma) *  distance

    logsigma = -2 * tf.log(sigma)

    coeff = logsigma - tf.log(2*tf.cast(np.pi, dtype=tf.float64))

    mat = mat + tf.transpose(coeff)

    return mat
    '''

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1
    # log_pi: K X 1
    # D: N X K

    # Outputs:
    # log Gaussian PDF N X K

    # 2.1.1 Log PDF for cluster k

    # compute pairwise squared distance as in part 1.1

    na = tf.reduce_sum(tf.square(X), 1)
    nb = tf.reduce_sum(tf.square(mu), 1)

    # na is N x K, nb is N x K
    # broadcast na along columns, broadcast nb along rows
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidean difference matrix
    D = na - 2 * tf.matmul(X, mu, False, True) + nb  # N x K

    sigma2_mat = tf.ones([tf.shape(X)[0], 1], dtype=tf.float64) * tf.reshape(tf.square(sigma), [1,
                                                                                                -1])  # square to get variances, then reshape into row tensor

    coeff_1 = -(tf.shape(X)[0] / 2) * tf.log(2 * tf.cast(np.pi, dtype=tf.float64))  # 1 X 1

    coeff_2 = -(1 / 2) * tf.log(
        tf.pow(sigma, tf.cast(tf.fill([tf.shape(sigma)[0], 1], tf.shape(X)[1]), dtype=tf.float64)))  # K x 1

    coeff = coeff_1 + coeff_2

    mat = -(1 / 2) * tf.multiply(tf.math.reciprocal(sigma2_mat), D)

    return tf.squeeze(coeff) + mat

'''

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # TODO
    coeff = tf.squeeze(log_pi) + log_PDF
    return coeff - tf.expand_dims(hlp.reduce_logsumexp(coeff), 1)



if __name__ == "__main__":

    # Loading data
    # data = np.load('data100D.npy')
    data = np.load('data2D.npy')
    [num_pts, dim] = np.shape(data)
    # print("hello")

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
    K = 5

    x = tf.placeholder(name = 'x', dtype = tf.float64, shape = (None, data.shape[1]))
    mu = tf.get_variable(name = 'mean', dtype=tf.float64, shape=(K, data.shape[1]),
                         initializer=tf.initializers.random_normal(seed=0))

    stdvec = tf.get_variable(name = 'std', dtype=tf.float64, shape=(K, 1),
                         initializer=tf.initializers.random_normal(seed=0))

    prior = tf.get_variable(name = 'pi', dtype=tf.float64, shape=(K, 1),
                         initializer=tf.initializers.random_normal(seed=0))

    sigma = tf.exp(prior)
    log_gauss_pdf = log_GaussPDF(x, mu, sigma)
    log_prior = hlp.logsoftmax(prior) #To ensure that priors are normalized
    log_post = log_posterior(log_gauss_pdf, log_prior)

    #Defining loss function
    temp = hlp.reduce_logsumexp(tf.squeeze(log_prior) + log_gauss_pdf)
    loss = -tf.reduce_sum(temp)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_loss = []

        for epoch in range(epochs):
            sess.run(optimizer, feed_dict={x: data})
            trainingloss = sess.run(loss, feed_dict={x: data})
            train_loss.append(trainingloss)

            print("Training Loss: {}".format(trainingloss))

        mu_after_training = sess.run(mu)
        sigma_after_training = sess.run(sigma)
        posterior_after_training = sess.run(log_post, feed_dict={x: data})

        print("Final mu: {}".format(mu_after_training))

        plt.plot(train_loss)
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')

        plt.title("Training Loss, K={}".format(K))
        plt.savefig("Loss{}Clusters".format(K))
        plt.show()

        #####CLUSTERS - CHANGE THIS

        final_mu = mu_after_training
        final_posterior = posterior_after_training


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






