#################################
# Your name: Itamar Shor , 315129551
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x_samples = np.random.uniform(0, 1, size=m)
        x_samples.sort()
        y_samples = []
        for x in x_samples:
            x_in_first_range = 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1
            y = np.random.choice([0, 1], size=1, p=[0.2, 0.8])[0] if x_in_first_range else \
                np.random.choice([0, 1], size=1, p=[0.9, 0.1])[0]
            y_samples.append(y)
        return np.column_stack((x_samples, y_samples))

    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """
        s = self.sample_from_D(m)
        x = s[:, 0]  # getting s first column
        y = s[:, 1]  # getting s second column
        plt.scatter(x, y, color='blue')
        plt.ylim((-0.1, 1.1))
        for l in [0.2, 0.4, 0.6, 0.8]:
            plt.axvline(x=l, color='green', linestyle='--')
        predictor = intervals.find_best_interval(x, y, k)[0]
        for interval in predictor:
            plt.axhline(y=1.05, color='red', xmin=interval[0], xmax=interval[1], linewidth=2)

        #  plt.show()

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        true_errors = []
        empirical_errors = []
        m_list = []  # x axis
        for m in range(m_first, m_last + 1, step):
            m_list.append(m)
            true_errors_helper = []
            empirical_errors_helper = []
            for i in range(T):
                s = self.sample_from_D(m)
                x = s[:, 0]  # getting s first column
                y = s[:, 1]  # getting s second column
                predictor, nof_miss = intervals.find_best_interval(x, y, k)
                empirical_errors_helper.append(nof_miss / m)
                true_errors_helper.append(self.calc_true_error(predictor))
            empirical_errors.append(self.mean(empirical_errors_helper))
            true_errors.append(self.mean(true_errors_helper))

        plt.plot(m_list, empirical_errors)
        plt.plot(m_list, true_errors)
        # plt.show()
        return np.column_stack((empirical_errors, true_errors))

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        s = self.sample_from_D(m)
        x = s[:, 0]  # getting s first column
        y = s[:, 1]  # getting s second column
        true_errors = []
        empirical_errors = []
        min_index = 0
        min_error = 1000
        k_list = []  # x axis
        for k in range(k_first, k_last + 1, step):
            k_list.append(k)
            predictor, nof_miss = intervals.find_best_interval(x, y, k)
            true_errors.append(self.calc_true_error(predictor))
            empirical_errors.append(nof_miss / m)
            min_error = (nof_miss / m) if min_error > (nof_miss / m) else min_error
            min_index = k if (nof_miss / m) == min_error else min_index

        plt.plot(k_list, empirical_errors)
        plt.plot(k_list, true_errors)
        # plt.show()
        # print("min index is", min_index)
        return min_index

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """

        s = self.sample_from_D(m)
        x = s[:, 0]  # getting s first column
        y = s[:, 1]  # getting s second column
        true_errors = []
        empirical_errors = []
        penalties = []
        penalty_with_empirical = []
        min_index = 0
        min_error = 1000
        k_list = []  # x axis
        for k in range(k_first, k_last + 1, step):
            k_list.append(k)
            predictor, nof_miss = intervals.find_best_interval(x, y, k)
            true_errors.append(self.calc_true_error(predictor))
            empirical_errors.append(nof_miss / m)
            penalty_temp = self.calc_penalty(m, k, 0.1)
            penalties.append(penalty_temp)
            penalty_with_empirical.append(penalty_temp + nof_miss / m)
            min_error = (penalty_temp + nof_miss / m) if min_error > (penalty_temp + nof_miss / m) else min_error
            min_index = k if (penalty_temp + nof_miss / m) == min_error else min_index

        plt.plot(k_list, empirical_errors, label='empirical error')
        plt.plot(k_list, true_errors, label='true error')
        plt.plot(k_list, penalties, label='penalty')
        plt.plot(k_list, penalty_with_empirical, label='penalty with empirical error')
        plt.legend(loc='upper right')
        # plt.show()
        # print("min index is", min_index)
        return min_index

    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        results = []
        for i in range(T):
            s = self.sample_from_D(m)
            np.random.shuffle(s)  # so we don't get the highest 20% of the sample for the test
            train = s[:int(0.8 * m)]
            train = sorted(train, key=lambda elem: elem[0])
            train = np.asarray(train)
            train_x = train[:, 0]
            train_y = train[:, 1]
            holdout = s[int(0.8 * m):]  # there is no need to sort the test elements
            holdout_x = holdout[:, 0]
            holdout_y = holdout[:, 1]
            min_index = 0
            min_error = 1000
            min_error_predictor = []
            for k in range(1, 10 + 1):
                predictor, nof_miss = intervals.find_best_interval(train_x, train_y, k)
                test_error = self.calc_error_for_holdout_set(predictor, holdout_x, holdout_y)
                min_error = test_error if min_error > test_error else min_error
                min_index = k if test_error == min_error else min_index
                min_error_predictor = predictor if test_error == min_error else min_error_predictor
            results.append((min_index, min_error, min_error_predictor))
        k_min = min(results, key=lambda elem: elem[1])[0]
        best_predictor = min(results, key=lambda elem: elem[1])[2]
        # print("best predictor is ", best_predictor)
        # print("min index is", k_min)
        return k_min

    #################################
    # Place for additional methods
    def mean(self, l):
        return sum(l) / len(l)

    def calc_true_error(self, intervals_list):
        """" calculate the expectation of the error of the hypothesis
        """
        # calculate the size of [x,y]  with [a,b]
        intersect = lambda x, y, a, b: min(y, b) - max(x, a)
        E = 0
        for fixed_interval in [(0, 0.2, 0.8), (0.2, 0.4, 0.1), (0.4, 0.6, 0.8), (0.6, 0.8, 0.1), (0.8, 1, 0.8)]:
            intersect_sum = 0
            for interval in intervals_list:
                p = fixed_interval[2]  # P[Y=1|x in fixed_interval]
                curr_intersect = intersect(fixed_interval[0], fixed_interval[1], interval[0], interval[1])
                if curr_intersect > 0:  # if the intervals aren't disjoint
                    intersect_sum += curr_intersect
            E += (1 - p) * intersect_sum + p * (0.2 - intersect_sum)
        return E

    def calc_penalty(self, m, k, delta):
        """" calculate the penalty function according to:
            m - number of samples
            k - hypothesis class complexity
            delta - error upper bound
        """
        pow_expression = (m * np.e) / k
        ln_expression = (4 / delta) * (np.power(pow_expression, 2 * k))
        return np.sqrt((8 / m) * np.log(ln_expression))

    def calc_error_for_holdout_set(self, intervals_list, test_x, test_y):
        """" calculate the empirical error of the predictor on the holdout set
        """
        error_cnt = 0
        for i in range(len(test_x)):
            x_label = 0
            for interval in intervals_list:
                if interval[0] <= test_x[i] <= interval[1]:
                    x_label = 1
            error_cnt = error_cnt + 1 if (x_label != test_y[i]) else error_cnt
        return error_cnt / (len(test_x))
    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.draw_sample_intervals(100, 3)
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500, 3)
