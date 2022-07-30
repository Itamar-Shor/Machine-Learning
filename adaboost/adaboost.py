#################################
# Your name: Itamar Shor 315129551
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    n = len(X_train)
    D = [1 / n for i in range(n)]  # init distribution
    hypotheses = []
    weights = []
    for t in range(T):
        hypothesis, train_error = get_WL(D, X_train, y_train)
        hypotheses.append(hypothesis)
        weight = 0.5 * np.log((1 - train_error) / train_error)
        weights.append(weight)
        D = update_distribution(X_train, y_train, D, weight, train_error, hypothesis)
    return hypotheses, weights


##############################################
# You can add more methods here, if needed.

predict = lambda hypothesis, x: hypothesis[0] if (x <= hypothesis[2]) else -hypothesis[0]


def get_WL(D, X_train, y_train):
    """
    returns:
        the best weak learner. weak learner is of type:
            - type 1: -1 iff x <= th
            - type 2: 1 iff x <= th
    """
    # type 1 --> 1 if x >= th
    best_th_type_1 = -1
    best_word_index_type_1 = -1
    min_error_type_1 = 100
    # type 2 --> 1 if x <= th
    best_th_type_2 = -1
    best_word_index_type_2 = -1
    min_error_type_2 = 100

    n = len(X_train)
    for word_idx in range(len(X_train[0])):
        data_per_word = [(i, X_train[i][word_idx]) for i in range(n)]
        data_per_word.sort(key=lambda x: x[1])  # sort by word count
        min_th = (int)(data_per_word[0][1])
        max_th = (int)(data_per_word[n - 1][1])
        th_dict_label_1 = {th: 0 for th in range(min_th, max_th + 1)}
        th_dict_label_minus_1 = {th: 0 for th in range(min_th, max_th + 1)}
        nof_1, nof_minus_1 = 0, 0
        for i in range(n):
            curr_th = (int)(data_per_word[i][1])
            if y_train[data_per_word[i][0]] == 1:
                th_dict_label_1[curr_th] += D[data_per_word[i][0]]
                nof_1 += D[data_per_word[i][0]]
            else:
                th_dict_label_minus_1[curr_th] += D[data_per_word[i][0]]
                nof_minus_1 += D[data_per_word[i][0]]

        curr_error_type_1 = nof_minus_1  # classify all as 1 (mistake on all -1 instances)
        if curr_error_type_1 < min_error_type_1:
            min_error_type_1 = curr_error_type_1
            best_word_index_type_1 = word_idx
            best_th_type_1 = min_th - 1

        curr_error_type_2 = nof_1  # classify all as -1 (mistake on all 1 instances)
        if curr_error_type_2 < min_error_type_2:
            min_error_type_2 = curr_error_type_2
            best_word_index_type_2 = word_idx
            best_th_type_2 = min_th - 1

        for th in range(min_th, max_th + 1):
            curr_error_type_1 += (th_dict_label_1[th] - th_dict_label_minus_1[th])
            if curr_error_type_1 < min_error_type_1:
                min_error_type_1 = curr_error_type_1
                best_word_index_type_1 = word_idx
                best_th_type_1 = th

            curr_error_type_2 += (-th_dict_label_1[th] + th_dict_label_minus_1[th])
            if curr_error_type_2 < min_error_type_2:
                min_error_type_2 = curr_error_type_2
                best_word_index_type_2 = word_idx
                best_th_type_2 = th

    if min_error_type_1 < min_error_type_2:
        return (-1, best_word_index_type_1, best_th_type_1), min_error_type_1
    return (1, best_word_index_type_2, best_th_type_2), min_error_type_2


def update_distribution(X_train, y_train, D, weight, train_error, hypothesis):
    """
    returns:
        update the distribution to the next adaboost iteration.
    """
    new_D = []
    denominator = 2 * np.sqrt((1 - train_error) * train_error)  # according to Q1.a in the Theory Questions
    for idx in range(len(X_train)):
        curr = D[idx] * (np.exp(-weight * y_train[idx] * predict(hypothesis, X_train[idx][hypothesis[1]]) ))
        new_D.append(curr / denominator)
    return new_D


def calc_final_hypothesis_error(hypotheses, alpha_vals, X, y, t):
    error = 0
    for idx in range(len(X)):
        estimation = sum([alpha_vals[i] * predict(hypotheses[i], X[idx][hypotheses[i][1]]) for i in range(t)])
        if (estimation * y[idx]) < 0:
            error += 1
    return error / len(X)


def calc_final_hypothesis_exp_loss(hypotheses, alpha_vals, X, y, t):
    loss = 0
    for idx in range(len(X)):
        power = -y[idx] * sum([alpha_vals[i] * predict(hypotheses[i], X[idx][hypotheses[i][1]]) for i in range(t)])
        exp = np.exp(power)
        loss += exp
    return loss / len(X)


def plot_section_a(hypotheses, alpha_vals, X_train, y_train, X_test, y_test, T):
    """
    plot:
        the training error and test error of the classifier at each iteration
    """
    t = []
    test_error = []
    train_error = []
    for idx in range(1, T + 1):
        t.append(idx - 1)
        test_error.append(calc_final_hypothesis_error(hypotheses, alpha_vals, X_test, y_test, idx))
        train_error.append(calc_final_hypothesis_error(hypotheses, alpha_vals, X_train, y_train, idx))
    plt.plot(t, train_error, color='blue', label='train error')
    plt.plot(t, test_error, color='green', label='test error')
    plt.legend()
    plt.show()


def section_b(X_train, y_train, vocab, T=10):
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    print("The following weak learners were chosen by adaboost:")
    for hypothesis in hypotheses:
        print(hypothesis, "matching word = ", vocab[hypothesis[1]])


def plot_section_c(hypotheses, alpha_vals, X_train, y_train, X_test, y_test, T):
    """
    plot:
        the training error and test error of the classifier at each iteration
    """
    t = []
    test_error = []
    train_error = []
    for idx in range(1, T + 1):
        t.append(idx - 1)
        test_error.append(calc_final_hypothesis_exp_loss(hypotheses, alpha_vals, X_test, y_test, idx))
        train_error.append(calc_final_hypothesis_exp_loss(hypotheses, alpha_vals, X_train, y_train, idx))
    plt.plot(t, train_error, color='blue', label='train error')
    plt.plot(t, test_error, color='green', label='test error')
    plt.legend()
    plt.show()


##############################################


def main():
    data = parse_data()
    if not data:
        return

    (X_train, y_train, X_test, y_test, vocab) = data
    T = 80
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    plot_section_a(hypotheses, alpha_vals, X_train, y_train, X_test, y_test, T)
    section_b(X_train, y_train, vocab)
    plot_section_c(hypotheses, alpha_vals, X_train, y_train, X_test, y_test, T)


if __name__ == '__main__':
    main()
