from sklearn.datasets import fetch_openml
import numpy.random
import matplotlib.pyplot as plt
from scipy.spatial import distance

mnist = fetch_openml('mnist_784')
data = mnist['data']
labels = mnist['target']

idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]


def k_nn(image_set, labels_vec, query_image, k):
    # array of tuples (idx,distance of the query_img from image_set[idx])
    distances = [(i, distance.euclidean(query_image, image_set[i])) for i in range(len(image_set))]
    distances.sort(key=lambda tup: tup[1])
    neighbors_labels_counter = [0 for i in range(10)]
    chosen_label = -1
    curr_max_cnt = 0
    for i in range(k):
        curr_label = int(labels_vec[distances[i][0]])
        neighbors_labels_counter[curr_label] += 1
        if curr_max_cnt < neighbors_labels_counter[curr_label]:
            curr_max_cnt = neighbors_labels_counter[curr_label]
            chosen_label = curr_label
    return chosen_label


def check_result_with_0_1_loss(n, k):
    success_counter = 0
    for i in range(len(test)):
        predictor_label = k_nn(train[:n], train_labels[:n], test[i], k)
        if int(test_labels[i]) == predictor_label:
            success_counter += 1
    return success_counter / (len(test))


def test_with_increasing_n(n_min, n_max, k=1, jumps=1):
    results = []
    for n in range(n_min, n_max + 1, jumps):
        results.append(check_result_with_0_1_loss(n, k))
    n = [i for i in range(n_min, n_max + 1, jumps)]
    plt.plot(n, results, color='lightblue', linewidth=3)
    plt.xlabel('n')
    plt.ylabel('score')
    plt.show()
    return


def test_with_increasing_k(k_min, k_max, n=1000, jumps=1):
    results = []
    for k in range(k_min, k_max + 1, jumps):
        results.append(check_result_with_0_1_loss(n, k))
    k = [i for i in range(k_min, k_max + 1, jumps)]
    plt.plot(k, results, color='lightblue', linewidth=3)
    plt.xlabel('k')
    plt.ylabel('score')
    plt.show()
    return


print("start")
print(check_result_with_0_1_loss(1000, 10))
test_with_increasing_k(1, 100)
test_with_increasing_n(100, 5000, jumps=100)
print("done")
