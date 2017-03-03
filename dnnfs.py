# import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def softmax(images, weights):
    """
    :param images: samples x dimension input images matrix
    :param weights: dimension x classes weights matrix
    :return: samples x classes matrix
    """
    z = images.dot(weights)  # samples x classes
    z -= np.max(z, axis=1, keepdims=True)
    ans = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    return ans


def J(w, images, labels, alpha=0.):
    """
    :param w: dimensions x classes weight matrix
    :param images: samples x dimensions images matrix
    :param labels: samples x classes target matrix
    :param alpha: regularization term
    :return:
    """
    x = images
    y = labels
    m = x.shape[0]
    small_constant = 0
    pred = softmax(x, w)
    cost_mat = np.multiply(y, np.log(pred + small_constant))
    cost = (-1. / m) * np.sum(np.sum(cost_mat, axis=1))
    cost += (alpha/(2*m)) * np.linalg.norm(w)
    return cost


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
    grad = (1./m) * np.dot((pred - y).T, x)
    grad += (alpha/m) * w.T
    return grad.T


def gradientDescent(trainingimages, trainingLabels, alpha=0.):
    x = trainingimages
    y = trainingLabels
    dimensions = x.shape[1]
    classes = y.shape[1]
    cost_history = np.array([])
    epsilon = 0.9
    #w = np.random.rand(dimensions, classes)
    w = np.zeros((dimensions, classes))
    iterations = 300

    for i in xrange(iterations):
        newgrad = gradJ(w, x, y, alpha)
        w -= (epsilon * newgrad)
        cost = J(w, x, y, alpha)
        cost_history = np.append(cost_history, cost)
        if i % 10 == 0:
            print "Iteration: ", i, " Cost: ", cost, "||w|| :", np.linalg.norm(w)

    plt.plot(np.linspace(1, iterations, iterations), cost_history, label="Training Cost")
    plt.legend()
    plt.ylabel('Training Cost')
    plt.xlabel('Iterations')
    plt.title("Cross-entropy loss values")
    plt.show()
    return w


def reportCosts(w, trainingimages, trainingLabels, testingimages, testingLabels, alpha=0.):
    print "Training cost: {}".format(J(w, trainingimages, trainingLabels, alpha))
    print "Testing cost:  {}".format(J(w, testingimages, testingLabels, alpha))


def report_accuracy(images, labels):
    acc = np.mean(np.argmax(images.dot(w), axis=1) == np.argmax(labels, axis=1))
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
    w = gradientDescent(trainingImages, trainingLabels, alpha)
    reportCosts(w, trainingImages, trainingLabels, testingImages, testingLabels)
    print "Accuracy is", report_accuracy(testingImages, testingLabels), "%"
    dt = int(time.time() - start)
    print("Execution time %d sec" % dt)

    # testw = np.ndarray.flatten(w)
    # from scipy.optimize import check_grad
    # print "Check grad value is ", check_grad(J, gradJ, testw, trainingImages, trainingLabels, 1e2)