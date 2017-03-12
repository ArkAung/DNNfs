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


def stochastic_J(y_hat, images, labels, alpha=0.):
    x = images
    y = labels
    m = x.shape[0]
    cost_mat = np.multiply(y, np.log(y_hat))
    cost = (-1. / m) * np.sum(np.sum(cost_mat, axis=1))
    #     cost += (alpha / (2 * m)) * np.linalg.norm(w)
    return cost


def J(w1, w2, b1, b2, images, labels,
      alpha=0.):  # TODO: Temp scaffold, remove this later on merge with the above function
    x = images
    y = labels
    m = x.shape[0]
    h1, y_hat = feedforward(w1, w2, b1, b2, x, y)
    cost_mat = np.multiply(y, np.log(y_hat))
    cost = (-1. / m) * np.sum(np.sum(cost_mat, axis=1))
    #     cost += (alpha / (2 * m)) * np.linalg.norm(w)
    return cost


def feedforward(w1, w2, b1, b2, images, labels, aplha=.0):
    x = images
    h1 = relu(x.dot(w1.T) + b1)
    y_hat = softmax(h1.dot(w2.T) + b2)
    return h1, y_hat


def grad_layer2(h1, y_hat, images, labels, alpha=0.):
    x = images
    y = labels
    h2 = (y_hat - y)
    dJ_dw2 = h2.T.dot(h1)
    dJ_b2 = h2
    return dJ_dw2, dJ_b2


def grad_layer1(h1, y_hat, w_1, w_2, images, labels, alpha=0.):
    x = images
    y = labels
    dJ_dh1 = (y_hat - y).dot(w_2)
    g = dJ_dh1 * relu_prime(x.dot(w_1.T))
    dJ_dw1 = g.T.dot(x)
    dJ_db1 = g
    return dJ_dw1, dJ_db1


def gradientDescent(trainingimages, trainingLabels, h_nodes, epsilon, batch_size, epochs, alpha=0., searching=False):
    x = trainingimages
    y = trainingLabels
    dimensions = x.shape[1]
    classes = y.shape[1]
    sample_size = x.shape[0]
    cost_history = np.array([])
    batch_history = np.array([])

    mu, sigma = 0, 0.1
    w1 = np.random.normal(mu, sigma, (h_nodes, dimensions))
    b1 = np.ones((1, h_nodes))
    w2 = np.random.normal(mu, sigma, (classes, h_nodes))
    b2 = np.ones((1, classes))

    num_batches = sample_size / batch_size
    for e in xrange(epochs):
        x_y = np.append(x, y, axis=1)
        np.random.shuffle(x_y)
        x_s = x_y[:, :dimensions]
        y_s = x_y[:, dimensions:]
        for i in xrange(num_batches):
            start = i * batch_size
            end = start + batch_size
            x_batch = x_s[start:end]
            y_batch = y_s[start:end]
            h1, y_hat = feedforward(w1, w2, b1, b2, x_batch, y_batch)  # Do feedforward pass
            gradw2, gradb2 = grad_layer2(h1, y_hat, x_batch, y_batch)
            w2 -= (epsilon * gradw2)
            b2 -= (epsilon * np.sum(gradb2, axis=0, keepdims=True))
            gradw1, gradb1 = grad_layer1(h1, y_hat, w1, w2, x_batch, y_batch)
            w1 -= (epsilon * gradw1)
            b1 -= (epsilon * np.sum(gradb1, axis=0, keepdims=True))

            cost = stochastic_J(y_hat, x_batch, y_batch, alpha)
            batch_history = np.append(batch_history, cost)

        cost_history = np.append(cost_history, cost)
        batch_acc = report_accuracy(w1, w2, b1, b2, validationImages, validationLabels)
        if e % 2 == 0:
            # print "Epochs: ", e, "Cost: ", cost, " Validation acc: ", batch_acc, "||w1|| :", np.linalg.norm(w1), "||w2|| :", np.linalg.norm(w2)
            print("Epochs: %d Cost: %.5f Validation Acc: %.2f ||w1||: %.5f ||w2||: %.5f" % (
                e,cost,batch_acc,np.linalg.norm(w1), np.linalg.norm(w2)))

    if (searching):
        return cost, batch_acc

    plt.plot(np.linspace(1, epochs * num_batches, epochs * num_batches), batch_history, label="Training Cost")
    plt.legend()
    plt.ylabel('Training Cost')
    plt.xlabel('Epochs * batches')
    plt.title("Cross-entropy loss values")
    plt.show()
    plt.plot(np.linspace(1, epochs, epochs), cost_history, label="Training Cost")
    plt.legend()
    plt.ylabel('Training Cost')
    plt.xlabel('Epochs')
    plt.title("Cross-entropy loss values")
    plt.show()
    return w1, w2, b1, b2


def reportCosts(w1, w2, b1, b2, trainImg , trainLbl, valiImg, valiLbl, testImg, testLbl, alpha=0.):
    print "Training cost: {}".format(J(w1, w2, b1, b2, trainImg, trainLbl, alpha))
    print "Validation cost:  {}".format(J(w1, w2, b1, b2, valiImg, valiLbl, alpha))
    print "Testing cost:  {}".format(J(w1, w2, b1, b2, testImg, testLbl, alpha))


def report_accuracy(w1, w2, b1, b2, images, labels):
    h1, y_hat = feedforward(w1, w2, b1, b2, images, labels)
    acc = np.mean(np.argmax(y_hat, axis=1) == np.argmax(labels, axis=1))
    return acc * 100


def predict(images, labels, w1, w2, b1, b2):
    h1, y_hat = feedforward(w1, w2, b1, b2, images, labels)
    predicted = np.argmax(y_hat)
    real = np.argmax(labels)
    return predicted, real

def findBestHyperparameters():
    h_nodes = [20, 20, 20, 30, 30, 30, 40, 40, 20, 30]
    l_rate = [1e-4, 1e-4, 1e-5, 0.0005, 0.00002, 0.00001, 0.0006, 0.0006, 0.00007, 0.0007]
    b_size = [64, 32, 64, 128, 512, 256, 16, 64, 36, 16]
    epochs = 11
    min_cost = 100
    max_acc = 0
    best = 0
    for i in range(10):
        print ("Trying parameters - Hidden Nodes: %d Learning Rate: %.6f Batch Size: %d Epochs: %d" % (
            h_nodes[i], l_rate[i], b_size[i], epochs))
        cost, acc = gradientDescent(trainingImages, trainingLabels, h_nodes[i], l_rate[i], b_size[i], epochs, alpha, searching=True)
        if cost < min_cost:
            min_cost = cost
            best = i
        if acc > max_acc:
            max_acc = acc
            best = i
        print ("Current best parameters - Hidden Nodes: %d Learning Rate: %.6f Batch Size: %d Epochs: %d" % (
            h_nodes[best], l_rate[best], b_size[best], epochs))
    return h_nodes[i], l_rate[i], b_size[i], 50

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
    h_nodes, l_rate, b_size, epochs = findBestHyperparameters()
    print ("Best parameters - Hidden Nodes: %d Learning Rate: %.6f Batch Size: %d Epochs: %d" % (h_nodes, l_rate, b_size, epochs))
    w1, w2, b1, b2 = gradientDescent(trainingImages, trainingLabels, h_nodes, l_rate, b_size, epochs, alpha)

    dt = int(time.time() - start)
    print("Execution time %d sec" % dt)

    reportCosts(w1, w2, b1, b2, trainingImages, trainingLabels, validationImages, validationLabels, testingImages, testingLabels)
    print "Accuracy on Validation set: ", report_accuracy(w1, w2, b1, b2, validationImages, validationLabels), "%"
    print "Accuracy on Testing set: ", report_accuracy(w1, w2, b1, b2, testingImages, testingLabels), "%"
