import math
import numpy as np
import pandas as pd
from array import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None,
                 info_gain=None, value=None, probability=None):
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        # for leaf node
        self.value = value
        self.probability = probability


class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2):
        self.root = None
        self.min_samples_split = min_samples_split

    def build_tree(self, dataset):
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)
        # split until stopping conditions are met
        if num_samples >= self.min_samples_split:
            # find the best split
            best_split, values_list = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"] > 0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"])
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"])
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        values_list = []
        max_info_gain = -float("inf")
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            for poss_value in feature_values:
                # FOUND NAN VALUE
                if math.isnan(poss_value):
                    # check for possible values of feature
                    for check_poss_val in feature_values:
                        if not math.isnan(check_poss_val):
                            if len(values_list) == 0:
                                values_list.append([feature_index, check_poss_val, 1])
                            found = False
                            for i in range(len(values_list)):
                                if not found:
                                    found = False
                                    if values_list[i][0] == feature_index:
                                        if values_list[i][1] == check_poss_val:
                                            values_list[i][2] = values_list[i][2] + 1
                                            found = True
                                    if not found:
                                        # feature_index, poss_value, frequency
                                        values_list.append([feature_index, check_poss_val, 1])
                                        found = True

            # return the unique values of particular feature
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right, probability_left, probability_right = self.split(dataset, feature_index,
                                                                                              threshold, values_list)
                # check if children are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]

                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["probability_dataset_left"] = 1
                        best_split["dataset_right"] = dataset_right
                        best_split["probability_dataset_right"] = 1
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split, values_list

    def split(self, dataset, feature_index, threshold, values_list):
        values = {}
        count = 0
        probability_left = 1
        probability_right = 1
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])

        for i in range(len(values_list)):
            while values_list[i][0] == feature_index:
                # total cases for feature
                count = count + values_list[i][2]
                value_of_feature = values_list[i][1]
                values[value_of_feature] = values_list[i][2]  # frequency

        for i in range(len(values_list)):
            while values_list[i][0] == feature_index:
                value_of_feature = values_list[i][1]
                values[value_of_feature] = (values[value_of_feature]) / count  # probability

        for i in range(len(values_list)):
            while values_list[i][0] == feature_index:
                for j in range(len(values)):
                    if j <= threshold:
                        probability_left = probability_left * values[value_of_feature]
                probability_right = 1 - probability_left
        return dataset_left, dataset_right, probability_left, probability_right

    def information_gain(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)

    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "Info Gain: ", round(tree.info_gain, 3))
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions

    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)


def testing(X, y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0)
    classifier = DecisionTreeClassifier(min_samples_split=3)
    classifier.fit(X_train, Y_train)
    # classifier.print_tree()
    Y_pred = classifier.predict(X_test)
    print("Accuracy without missing values:", round(accuracy_score(Y_test, Y_pred), 3))
    print()

    # random attributes deleted
    prob = array('f', [0, 0.1, 0.2, 0.5])
    for p in prob:
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0)
        for row_index_del in range(X_train.shape[0]):
            for col_index_del in range(X_train.shape[1]):
                if np.random.uniform() < p:
                    X_train[row_index_del, col_index_del] = np.nan
        classifier = DecisionTreeClassifier(min_samples_split=3)
        classifier.fit(X_train, Y_train)
        # classifier.print_tree()
        Y_pred = classifier.predict(X_test)
        print("Accuracy with probability of missing values", round(p, 2), ":", round(accuracy_score(Y_test, Y_pred), 3))
        print()


if __name__ == '__main__':
    # Glass Identification - 10 att + target class
    print("GLASS dataset")
    print()
    df_g = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
                       header=None)
    # target in last column
    Xg = df_g.iloc[:, :-1].values
    yg = df_g.iloc[:, -1].values.reshape(-1, 1)
    testing(Xg, yg)

    # Wine - 13 att + target class
    print("WINE dataset")
    print()
    df_w = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
                       header=None)
    # target in first column
    Xw = df_w.iloc[:, 1:].values
    yw = df_w.iloc[:, 0].values.reshape(-1, 1)
    testing(Xw, yw)

    # Iris - 4 att + target class
    print("IRIS dataset")
    print()
    df_i = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", skiprows=1,
                       header=None)
    # target in last column
    Xi = df_i.iloc[:, :-1].values
    yi = df_i.iloc[:, -1].values.reshape(-1, 1)
    testing(Xi, yi)

