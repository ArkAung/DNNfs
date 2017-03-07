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
    h1 = relu(x.dot(w1))
    y_hat = softmax(h1.dot(w2))
    return h1, y_hat


def gradJ_w1(w, g_w2, images, labels, alpha=0.):
    x = images
    y = labels
    g = g_w2 * relu_prime(x.dot(w)) #get the results calculated up to w2
    block_x = g_w2.reshape(g_w2.shape[0] * g_w2.shape[1],) #vectorize the w_2 matrix
    
    return None


def gradJ_w2(w, images, labels, alpha=0.):
    x = images
    y = labels
    y_hat = softmax(x,w)
    return (y_hat - y).dot(w.T)


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
    cost_history = np.array([])
    epsilon = 0.9

    w1 = np.random.rand(dimensions, h_nodes)
    w2 = np.random.rand(h_nodes, classes)

    # w = np.zeros((dimensions, classes))
    iterations = 300

    for i in xrange(iterations):
        gradw2 = gradJ_w2(w2, x, y, alpha)
        gradw1 = gradJ_w1(w1, gradw2, x, y, alpha)

        w1 -= (epsilon * gradw1)
        w2 -= (epsilon * gradw2)

        cost = J(w1, w2, x, y, alpha)
        cost_history = np.append(cost_history, cost)
        if i % 10 == 0:
            print "Iteration: ", i, " Cost: ", cost, "||w1|| :", np.linalg.norm(w1), "||w2|| :", np.linalg.norm(w2)

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

    h_nodes = 30
    import time
    start = time.time()
    alpha = 1e2
    w1, w2 = gradientDescent(trainingImages, trainingLabels, alpha)
#   reportCosts(w1, w2, trainingImages, trainingLabels, testingImages, testingLabels)
    # print "Accuracy is", report_accuracy(testingImages, testingLabels), "%"
    # dt = int(time.time() - start)
    # print("Execution time %d sec" % dt)

    # testw = np.ndarray.flatten(w)
    # from scipy.optimize import check_grad
    # print "Check grad value is ", check_grad(J, gradJ, testw, trainingImages, trainingLabels, 1e2)
