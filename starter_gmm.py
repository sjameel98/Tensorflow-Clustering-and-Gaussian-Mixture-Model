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

    #distance = distanceFunc(X, mu)
    #sigma = tf.transpose(sigma)
    #m = tf.cast(D, dtype=tf.float64) * tf.log(sigma*tf.sqrt(2*tf.cast(np.pi, dtype=tf.float64))) + 1/(2*sigma*sigma) * distance
    #return -m
    distance = distanceFunc(X, mu)

    invsigma = tf.math.reciprocal(sigma)

    invsigma = -1/2 * tf.square(invsigma)

    mat = tf.transpose(invsigma) *  distance

    logsigma = -tf.cast((tf.shape(X)[1]), dtype=tf.float64) * tf.log(sigma)

    coeff = logsigma - tf.cast((tf.shape(X)[1]/2), dtype=tf.float64) * tf.log(2*tf.cast(np.pi, dtype=tf.float64))

    mat = mat + tf.transpose(coeff)

    return mat


def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # TODO
    coeff = tf.squeeze(log_pi) + log_PDF
    return coeff - tf.expand_dims(hlp.reduce_logsumexp(coeff), 1)
    #den = tf.expand_dims(hlp.reduce_logsumexp(coeff), 1)
    #return coeff/den




if __name__ == "__main__":

    # Loading data
    data = np.load('data100D.npy')
    # data = np.load('data2D.npy')
    [num_pts, dim] = np.shape(data)
    # print("hello")

    val_data = data

    # For Validation set
    is_valid = False

    if is_valid:
        valid_batch = int(num_pts / 3.0)
        np.random.seed(45689)
        rnd_idx = np.arange(num_pts)
        np.random.shuffle(rnd_idx)
        val_data = data[rnd_idx[:valid_batch]]
        data = data[rnd_idx[valid_batch:]]

    epochs = 1000 #5000
    lr = 0.005
    K = 5

    x = tf.placeholder(name = 'x', dtype = tf.float64, shape = (None, data.shape[1]))
    mu = tf.get_variable(name = 'mean', dtype=tf.float64, shape=(K, data.shape[1]),
                         initializer=tf.initializers.random_normal(seed=18786708))

    stdvec = tf.get_variable(name = 'std', dtype=tf.float64, shape=(K, 1),
                         initializer=tf.initializers.random_normal(seed=366901768))

    prior = tf.get_variable(name = 'pi', dtype=tf.float64, shape=(K, 1),
                         initializer=tf.initializers.random_normal(seed=1566557))

    sigma = tf.exp(stdvec)
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
            validloss = sess.run(loss, feed_dict={x: val_data})
            train_loss.append(trainingloss)

            print('Training loss = {} | Validation loss = {}'.format(trainingloss, validloss))

        mu_after_training = sess.run(mu)
        sigma_after_training = sess.run(sigma)
        posterior_after_training = sess.run(log_post, feed_dict={x: data})

        print("Final mu: {}".format(mu_after_training))
        print("Final stdev: {}".format(sigma_after_training))

        plt.plot(train_loss)
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')

        plt.title("Training Loss, K={}".format(K))
        plt.savefig("LossGMM{}Clusters100D".format(K))
        plt.show()

        #####CLUSTERS - CHANGE THIS

        final_mu = mu_after_training
        final_posterior = posterior_after_training

        labels = np.argmax(final_posterior, axis=1)
        unique, counts = np.unique(labels, return_counts=True)
        dictionary = dict(zip(unique, counts))


        for clusters in range(K):

            try:
                p = dictionary[clusters]* 100 / data.shape[0]
                print('% of points belonging to cluster {} is: {}% '.format(clusters, p))
            except:
                print('% of points belonging to cluster {} is: 0% '.format(clusters))



'''
        x_mu, y_mu = final_mu.T


        plt.scatter(data[:, 0], data[:, 1], c=labels, label='Data Points', s=25, alpha=0.8, cmap='Dark2')

        mark = ['o', '+', 'h', 'd', '*', 'P']
        for i in range(K):
            plt.scatter(final_mu[i, 0], final_mu[i, 1], marker=mark[i], label='Cluster {} Mean'.format(i), s=100, c='k')

        #plt.scatter(data[:, 0], data[:, 1], c=labels, label='data', s=10, alpha=0.8)
        #plt.scatter(final_mu[:, 0], final_mu[:, 1], cmap='r', marker='X', label='means', c='r')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('GMM, k = {}'.format(K))
        plt.legend()
        plt.savefig("GMM_{}Clusters".format(K))
        plt.show()

'''




