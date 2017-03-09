# import matplotlib.pyplot as plt
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


def J(w1, w2, images, labels, alpha=0.):
    """
    :param w: dimensions x classes weight matrix
    :param images: samples x dimensions images matrix
    :param labels: samples x classes target matrix
    :param alpha: regularization term
    :return:
    """
    # x = images
    # y = labels
    # m = x.shape[0]
    # small_constant = 0
    # pred = softmax(x, w)
    # cost_mat = np.multiply(y, np.log(pred + small_constant))
    # cost = (-1. / m) * np.sum(np.sum(cost_mat, axis=1))
    # cost += (alpha / (2 * m)) * np.linalg.norm(w)
    return 0


def feedforward(w1, w2, images, labels, aplha=.0):
    x = images
    h1 = relu(x.dot(w1.T))
    y_hat = softmax(h1.dot(w2.T))
    return h1, y_hat


def gradJ_w2(h1, y_hat, images, labels, alpha=0.):
    x = images
    y = labels
    return (y_hat - y).T.dot(h1) # TODO: This is not right, fix this


def gradJ_w1(h1, y_hat, w_1, w_2, images, labels, alpha=0.):
    # TODO: Figure out whether the w_2 that you use here is updated one or not (Most likely to be the updated one). If it is so, update the w2 first in gradient desdcent
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
    x = trainingimages[1]
    y = trainingLabels[1]

    dimensions = x.shape[1]
    classes = y.shape[1]
    cost_history = np.array([])
    epsilon = 1e-5

    h_nodes = 30
    batch_size = 500

    mu, sigma = 0, 0.1
    w1 = np.random.normal(mu, sigma, (h_nodes, dimensions))
    w2 = np.random.normal(mu, sigma, (classes, h_nodes))

    # w = np.zeros((dimensions, classes))
    iterations = 10

    # # TODO: design the program so that the gradJ_wi is a single function and takes in a single argument which is an np-array of hidden layer outputs and y_hat output
    for i in xrange(iterations):
        h1, y_hat = feedforward(w1, w2, x, y)  # Do feedforward pass
        gradw2 = gradJ_w2(h1, y_hat, x, y)
        w2 -= (epsilon * gradw2)
        gradw1 = gradJ_w1(h1, y_hat, w1, w2, x, y)
        w1 -= (epsilon * gradw1)

        # cost = J(w1, w2, x, y, alpha)
        # cost_history = np.append(cost_history, cost)
        # if i % 10 == 0:
        #     print "Iteration: ", i, " Cost: ", cost, "||w1|| :", np.linalg.norm(w1), "||w2|| :", np.linalg.norm(w2)
        if i % 2 == 0:
            print "Iteration: ", i, "||w1|| :", np.linalg.norm(w1), "||w2|| :", np.linalg.norm(w2)
    # plt.plot(np.linspace(1, iterations, iterations), cost_history, label="Training Cost")
    # plt.legend()
    # plt.ylabel('Training Cost')
    # plt.xlabel('Iterations')
    # plt.title("Cross-entropy loss values")
    # plt.show()
    return w1, w2


def reportCosts(w1, w2, trainingimages, trainingLabels, testingimages, testingLabels, alpha=0.):
    print "Training cost: {}".format(J(w1, w2, trainingimages, trainingLabels, alpha))
    print "Testing cost:  {}".format(J(w1, w2, testingimages, testingLabels, alpha))


# def report_accuracy(images, labels):
#     acc = np.mean(np.argmax(images.dot(w), axis=1) == np.argmax(labels, axis=1))
#     return acc * 100


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
    alpha = 0
    w1, w2 = gradientDescent(trainingImages, trainingLabels, alpha)
# reportCosts(w1, w2, trainingImages, trainingLabels, testingImages, testingLabels)
# print "Accuracy is", report_accuracy(testingImages, testingLabels), "%"
    dt = int(time.time() - start)
    print("Execution time %d sec" % dt)

# testw = np.ndarray.flatten(w)
# from scipy.optimize import check_grad
# print "Check grad value is ", check_grad(J, gradJ, testw, trainingImages, trainingLabels, 1e2)
