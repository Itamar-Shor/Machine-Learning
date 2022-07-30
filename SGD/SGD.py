#################################
# Your name: Itamar Shor, 315129551
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper_hinge():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def helper_ce():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = labels[train_idx[:6000]]

    validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
    validation_labels = labels[train_idx[6000:8000]]

    test_data_unscaled = data[8000 + test_idx, :].astype(float)
    test_labels = labels[8000 + test_idx]

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    """
    # each data sample is a 28*28 matrix (or 784 vector)
    w = [0 for i in range(28 * 28)]  # initialize w
    for i in range(1, T + 1):
        idx = np.random.randint(0, len(data))  # choosing x in random
        cond = ((np.dot(data[idx], w)) * labels[idx]) < 1
        eta_t = eta_0 / i
        calc_helper = np.multiply(1 - eta_t, w)
        if cond:
            w = np.add(np.multiply(eta_t * C * labels[idx], data[idx]), calc_helper)
        else:
            w = calc_helper
    return w


def SGD_ce(data, labels, eta_0, T):
    """
    Implements multi-class cross entropy loss using SGD.
    """
    Ws = [[0 for i in range(28 * 28)] for j in range(10)]  # classifier for every label
    for i in range(1, T + 1):
        idx = np.random.randint(0, len(data))  # choosing x in random
        classifier_gradients = calc_ce_gradients(data[idx], labels[idx], Ws)
        for j in range(10):
            Ws[j] = np.add(Ws[j], np.multiply((-1) * eta_0, classifier_gradients[j]))
    return Ws


#################################

# Place for additional code
# ============================== section 1: question (a) =======================================
def find_best_eta_hinge(train_data, train_labels, validation_data, validation_labels, T=1000, C=1):
    """
    Use cross-validation on the validation set to find the best eta
    """
    #eta_range = [np.float_power(10, i) for i in range(-5, 4)]
    eta_range = [1 + i / 10 for i in range(-9, 10)]
    res = []
    best_eta, best_accuracy = -1, -1
    for eta_0 in eta_range:
        avg = 0
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, C, eta_0, T)
            avg += calc_accuracy_rate_hinge(validation_data, validation_labels, w)
        avg = avg / 10
        res.append(avg)
        if best_accuracy < avg:
            best_accuracy = avg
            best_eta = eta_0

    plt.plot(eta_range, res)
    plt.xlabel('eta')
    plt.ylabel('average accuracy on validation set')
    plt.xscale('log')
    # plt.show()
    return best_eta


# ============================== section 1: question (b) =======================================
def find_best_C_hinge(train_data, train_labels, validation_data, validation_labels, eta_0, T=1000):
    """
    Use cross-validation on the validation set to find the best C
    """
    # C_range = [np.float_power(10, i, dtype=np.longdouble) for i in range(-5, 4)]
    C_range = [i / 100000 for i in range(1, 20, 2)]
    res = []
    best_C, best_accuracy = -1, -1
    for c in C_range:
        avg = 0
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, c, eta_0, T)
            avg += calc_accuracy_rate_hinge(validation_data, validation_labels, w)
        avg = avg / 10
        res.append(avg)
        if best_accuracy < avg:
            best_accuracy = avg
            best_C = c

    plt.plot(C_range, res)
    plt.xlabel('C')
    plt.ylabel('average accuracy on validation set')
    plt.xscale('log')
    # plt.show()
    return best_C


# ============================== section 1: question (c) =======================================
def train_classifier_hinge(train_data, train_labels, C, eta_0, T=20000):
    """
    Using the best C and eta, train the classifier
    """
    w = SGD_hinge(train_data, train_labels, C, eta_0, T)
    plt.imshow(np.reshape(w, (28, 28)))
    # plt.show()
    return w


# ============================== section 1: question (d) =======================================
def check_accuracy_of_estimator_hinge(test_data, test_labels, w):
    return calc_accuracy_rate_hinge(test_data, test_labels, w)


# ============================== section 2: question (a) =======================================
def find_best_eta_ce(train_data, train_labels, validation_data, validation_labels, T=1000):
    """
    Use cross-validation on the validation set to find the best eta
    """
    #eta_range = [np.float_power(10, i) for i in range(-7, 5)]
    eta_range = [np.float_power(10, -6) + i/10000000 for i in range(-5,5)]
    res = []
    best_eta, best_accuracy = -1, -1
    for eta_0 in eta_range:
        avg = 0
        for i in range(10):
            Ws = SGD_ce(train_data, train_labels, eta_0, T)
            avg += calc_accuracy_rate_ce(validation_data, validation_labels, Ws)
        avg = avg / 10
        res.append(avg)
        if best_accuracy < avg:
            best_accuracy = avg
            best_eta = eta_0

    plt.plot(eta_range, res)
    plt.xlabel('eta')
    plt.ylabel('average accuracy on validation set')
    plt.xscale('log')
    # plt.show()
    return best_eta


# ============================== section 2: question (b) =======================================
def train_classifier_ce(train_data, train_labels, eta_0, T=20000):
    """
    Using the best eta, train the classifier
    """
    Ws = SGD_ce(train_data, train_labels, eta_0, T)
    fig, axs = plt.subplots(2, 5)
    for row in range(2):
        for col in range(5):
            axs[row, col].imshow(np.reshape(Ws[5 * row + col], (28, 28)))
    # plt.show()
    return Ws


# ============================== section 2: question (c) =======================================
def check_accuracy_of_estimator_ce(test_data, test_labels, Ws):
    return calc_accuracy_rate_ce(test_data, test_labels, Ws)


def calc_accuracy_rate_hinge(data, labels, w):
    """
    checks the accuracy of the predictor w*x on 'data' (relative to hinge)
    """
    nof_correct_estimations = 0
    for i in range(len(data)):
        estimate_label = 1 if (np.dot(w, data[i]) >= 0) else -1
        if estimate_label == int(labels[i]):
            nof_correct_estimations += 1
    return nof_correct_estimations / len(data)


def calc_ce_gradients(data, label, Ws):
    """
    calculate l_CE gradient = (p(i | x;Ws) - I{i=label}) * data,
    where i is the idx of each classifier (Ws)
    """
    y = int(label)  # label is a string
    p = calc_p(data, Ws)
    p[y] -= 1  # substracting I{i=label}
    gradients = [np.multiply(i, data) for i in p]
    return gradients


def calc_p(data, Ws):
    """
    calculate p(i | x;Ws) = e^(W[label]*x) / (sum from 1 to 10 of e^(W[i]*x))
    """
    dot_products = [np.dot(data, w) for w in Ws]
    max_dot_products = np.max(dot_products)
    # adding the same constant (i.e max_dot_products) to all of the dot products doesn't change the gradient!
    # it used in order to prevent overflow
    es = [np.exp(item - max_dot_products) for item in dot_products]
    es_sum = np.sum(es)
    p = [i / es_sum for i in es]
    return p


def calc_accuracy_rate_ce(validation_data, validation_labels, Ws):
    """
    checks the accuracy of the predictor w*x on 'data' (relative to ce)
    """
    nof_correct_estimations = 0
    for j in range(len(validation_data)):
        max_product = np.dot(validation_data[j], Ws[0])  # initialize
        max_idx = 0
        for i in range(1, 10):
            current_product = np.dot(validation_data[j], Ws[i])
            if current_product > max_product:
                max_product = current_product
                max_idx = i
        if max_idx == int(validation_labels[j]):
            nof_correct_estimations += 1
    return nof_correct_estimations / len(validation_data)


#################################


# if __name__ == '__main__':
    # train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    # eta_hinge = find_best_eta_hinge(train_data, train_labels, validation_data, validation_labels)
    # print(eta_hinge)
    # C = find_best_C_hinge(train_data, train_labels, validation_data, validation_labels, eta_hinge)
    # print(C)
    # optimal_w = train_classifier_hinge(train_data, train_labels, C, eta_hinge)
    # accuracy_rate_hinge = check_accuracy_of_estimator_hinge(test_data, test_labels, optimal_w)
    # print(accuracy_rate_hinge)
    # train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
    # eta_ce = find_best_eta_ce(train_data, train_labels, validation_data, validation_labels)
    # print(eta_ce)
    # optimal_Ws = train_classifier_ce(train_data, train_labels, eta_ce)
    # accuracy_rate_ce = check_accuracy_of_estimator_ce(test_data, test_labels, optimal_Ws)
    # print(accuracy_rate_ce)
