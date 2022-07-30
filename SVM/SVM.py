#################################
# Your name: Itamar Shor , 315129551
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""


# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    linear_clf = svm.SVC(C=1000, kernel='linear')
    linear_clf.fit(X_train, y_train)
    create_plot(X_train, y_train, linear_clf)
    plt.title('linear kernel')
    # plt.show()
    quadratic_clf = svm.SVC(C=1000, kernel='poly', degree=2)
    quadratic_clf.fit(X_train, y_train)
    create_plot(X_train, y_train, quadratic_clf)
    plt.title('quadratic kernel')
    # plt.show()
    rbf_clf = svm.SVC(C=1000, kernel='rbf')
    rbf_clf.fit(X_train, y_train)
    create_plot(X_train, y_train, rbf_clf)
    plt.title('rbf kernel')
    # plt.show()
    return np.array([linear_clf.n_support_, quadratic_clf.n_support_, rbf_clf.n_support_])


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    val_res = []
    train_res = []
    C = [np.float_power(10, i) for i in range(-5, 6)]
    for c in C:
        linear_clf = svm.SVC(C=c, kernel='linear')
        linear_clf.fit(X_train, y_train)
        train_res.append(linear_clf.score(X_train, y_train))
        val_res.append(linear_clf.score(X_val, y_val))
        # create_plot(X_train, y_train, linear_clf)
        # plt.title('C = 10^%d' % np.log10(c))
        # plt.show()

    plt.plot(C, train_res, '-ok', label='training set accuracy')
    plt.legend()
    plt.xscale('log')
    # plt.show()
    plt.plot(C, val_res, '-ok', label='validation set accuracy')
    plt.legend()
    plt.xscale('log')
    # plt.show()
    return np.array(val_res)


def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    val_res = []
    train_res = []
    gamma_range = [np.float_power(10, i) for i in range(-5, 6)]
    # gamma_range = [i for i in range (1, 12)]
    for gamma in gamma_range:
        rbf_clf = svm.SVC(C=10, kernel='rbf', gamma=gamma)
        rbf_clf.fit(X_train, y_train)
        train_res.append(rbf_clf.score(X_train, y_train))
        val_res.append(rbf_clf.score(X_val, y_val))
        # create_plot(X_train, y_train, rbf_clf)
        # plt.title('gamma = 10^%d' % np.log10(gamma))
        # plt.show()

    plt.plot(gamma_range, train_res, '-ok', label='training set accuracy')
    plt.legend()
    plt.xscale('log')
    # plt.show()
    plt.plot(gamma_range, val_res, '-ok', label='validation set accuracy')
    plt.legend()
    plt.xscale('log')
    # plt.show()
    return np.array(val_res)


if __name__ == '__main__':
    train_x, train_y, val_x, val_y = get_points()
    print(train_three_kernels(train_x, train_y, val_x, val_y))
    print(linear_accuracy_per_C(train_x, train_y, val_x, val_y))
    print(rbf_accuracy_per_gamma(train_x, train_y, val_x, val_y))
