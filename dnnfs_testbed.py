% matplotlib
inline
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    ans = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    return ans


def relu(z):
    z[z < 0] = 0
    return z


def relu_prime(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z


def J(y_hat, images, labels, alpha=0.):
    x = images
    y = labels
    m = x.shape[0]
    cost_mat = np.multiply(y, np.log(y_hat))
    cost = (-1. / m) * np.sum(np.sum(cost_mat, axis=1))
    #     cost += (alpha / (2 * m)) * np.linalg.norm(w)
    return cost


def J2(w1, w2, images, labels, alpha=0.):
    x = images
    y = labels
    m = x.shape[0]
    h1, y_hat = feedforward(w1, w2, x, y)
    cost_mat = np.multiply(y, np.log(y_hat))
    cost = (-1. / m) * np.sum(np.sum(cost_mat, axis=1))
    #     cost += (alpha / (2 * m)) * np.linalg.norm(w)
    return cost


def feedforward(w1, w2, images, labels, aplha=.0):
    x = images
    h1 = relu(x.dot(w1.T))
    y_hat = softmax(h1.dot(w2.T))
    return h1, y_hat


def gradJ_w2(h1, y_hat, images, labels, alpha=0.):
    x = images
    y = labels
    return (y_hat - y).T.dot(h1)


def gradJ_w1(h1, y_hat, w_1, w_2, images, labels, alpha=0.):
    x = images
    y = labels
    dJ_dh1 = (y_hat - y).dot(w_2)
    g = dJ_dh1 * relu_prime(x.dot(w_1.T))
    # vec_input = x.reshape(x.shape[0] * x.shape[1], )
    return g.T.dot(x)


def gradJ(w, images, labels, alpha=0.):
    """
    :param w: dimension x classes weights matrix
    :param images: samples x dimensions images matrix
    :param labels: samples x classes target matrix
    :param alpha: regularization term
    :return: dimensions x classes matrix
    """
    x = images
    y = labels
    m = x.shape[0]

    pred = softmax(x, w)
    grad = (1. / m) * np.dot((pred - y).T, x)
    grad += (alpha / m) * w.T
    return grad.T


def gradientDescent(trainingimages, trainingLabels, alpha=0.):
    x = trainingimages
    y = trainingLabels
    dimensions = x.shape[1]
    classes = y.shape[1]
    sample_size = x.shape[0]
    cost_history = np.array([])
    epsilon = 1e-5

    batch_size = 500
    h_nodes = 20
    epochs = 50

    mu, sigma = 0, 0.1
    w1 = np.random.normal(mu, sigma, (h_nodes, dimensions))
    w2 = np.random.normal(mu, sigma, (classes, h_nodes))

    num_batches = sample_size / batch_size
    for e in xrange(epochs):
        x_y = np.append(x, y, axis=1)
        np.random.shuffle(x_y)
        x_s = x_y[:, :dimensions]
        y_s = x_y[:, dimensions:]
        for i in xrange(num_batches):
            start = i * num_batches
            end = start + (batch_size - 1)
            x_batch = x_s[start:end]
            y_batch = y_s[start:end]
            h1, y_hat = feedforward(w1, w2, x_batch, y_batch)  # Do feedforward pass
            gradw2 = gradJ_w2(h1, y_hat, x_batch, y_batch)
            w2 -= (epsilon * gradw2)
            gradw1 = gradJ_w1(h1, y_hat, w1, w2, x_batch, y_batch)
            w1 -= (epsilon * gradw1)

            cost = J(y_hat, x_batch, y_batch, alpha)
            cost_history = np.append(cost_history, cost)

        if e % 2 == 0:
            print "Epochs: ", e, "Cost: ", cost, "||w1|| :", np.linalg.norm(w1), "||w2|| :", np.linalg.norm(w2)

    plt.plot(np.linspace(1, epochs * num_batches, epochs * num_batches), cost_history, label="Training Cost")
    plt.legend()
    plt.ylabel('Training Cost')
    plt.xlabel('Epochs * Batches')
    plt.title("Cross-entropy loss values")
    plt.show()
    return w1, w2


def reportCosts(w1, w2, trainingimages, trainingLabels, testingimages, testingLabels, alpha=0.):
    print "Training cost: {}".format(J2(w1, w2, trainingimages, trainingLabels, alpha))
    print "Testing cost:  {}".format(J2(w1, w2, testingimages, testingLabels, alpha))


def report_accuracy(w1, w2, images, labels):
    h1, y_hat = feedforward(w1, w2, images, labels)
    acc = np.mean(np.argmax(y_hat, axis=1) == np.argmax(labels, axis=1))
    return acc * 100


def predict(image, label, weight):
    predicted = np.argmax(image.dot(weight))
    real = np.argmax(label)
    return predicted, real


if __name__ == "__main__":
    # Load data
    if ('trainingImages' not in globals()):
        trainingImages = np.load("datasets/mnist_train_images.npy")
        trainingLabels = np.load("datasets/mnist_train_labels.npy")
        validationImages = np.load("datasets/mnist_validation_images.npy")
        validationLabels = np.load("datasets/mnist_validation_labels.npy")
        testingImages = np.load("datasets/mnist_test_images.npy")
        testingLabels = np.load("datasets/mnist_test_labels.npy")

    import time

    start = time.time()
    alpha = 1e2
    w1, w2 = gradientDescent(trainingImages, trainingLabels, alpha)

    reportCosts(w1, w2, trainingImages, trainingLabels, validationImages, validationLabels)
    print "Accuracy is", report_accuracy(w1, w2, validationImages, validationLabels), "%"
    dt = int(time.time() - start)
    print("Execution time %d sec" % dt)

# testw = np.ndarray.flatten(w)
# from scipy.optimize import check_grad
# print "Check grad value is ", check_grad(J, gradJ, testw, trainingImages, trainingLabels, 1e2)
