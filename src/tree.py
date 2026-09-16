import numpy as np

from .loss import gini, entropy
from .node import Node

class DecisionTree:
    def __init__(self, impurity="gini", max_depth=None, min_split = 2):
        self.max_depth = max_depth
        self.min_split = min_split
        self.impurity = impurity
        return

    # single row predictor to avoid confusing recursion in main prediction function
    def _predict_row(self, x, node):
        # only leaf nodes have a value != None
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_row(x, node.left_node)
        else:
            return self._predict_row(x, node.right_node)

    # returns leaf prediction
    def predict(self, x):
        # go sample by sample
        predictions = []

        for i in range(x.shape[0]):
            predictions.append(self._predict_row(x[i,:], self.root))
        return np.array(predictions)

    # training, (calls build_tree, stores in root)
    def fit(self, x, y):
        self.root = self._build_tree(x, y, 0)
        return self.root

    def _impurity(self, y):
        if self.impurity == "gini":
            return gini(y)
        elif self.impurity == "entropy":
            return entropy(y)
        else:
            raise ValueError("unknown impurity")

    def _best_split(self, x, y):

        best_feature = None
        best_threshold = None
        best_impurity = np.inf
        # check each partition class proportions and loss
        for j in range(x.shape[1]):

            # construct unique thresholds for column j
            vals = np.unique(x[:,j])
            thresholds = (vals[:-1] + vals[1:]) / 2

            for threshold in thresholds:

                left_ind = x[:,j] <= threshold
                right_ind = x[:,j] > threshold

                y_left = y[left_ind]
                y_right = y[right_ind]

                n_left = len(y_left)
                n_right = len(y_right)
                n = n_left + n_right

                # ignore invalid splits
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                left_impurity = gini(y_left)
                right_impurity = gini(y_right)

                gini_imp = (n_left / n) * left_impurity + (n_right / n) * right_impurity

                if gini_imp < best_impurity:
                    best_impurity = gini_imp
                    best_feature = j
                    best_threshold = threshold
        return best_feature, best_threshold, best_impurity


    def _build_tree(self, x, y, depth):

        # check if node is pure
        if len(np.unique(y)) == 1:
            return Node(value = y[0])

        # check if max depth reached or node has fewer than minimum data requirement
        if depth == self.max_depth or x.shape[0] < self.min_split:
            classes, counts = np.unique(y, return_counts=True)
            return Node(value = classes[np.argmax(counts)])

        best_feature, best_threshold, best_impurity = self._best_split(x, y)

        # check if no valid threshold (all features are the same)
        if best_feature is None:
            classes, counts = np.unique(y, return_counts=True)
            return Node(value = classes[np.argmax(counts)])

        left_indices = x[:, best_feature] <= best_threshold
        right_indices = x[:, best_feature] > best_threshold
        return Node(feature=best_feature, threshold = best_threshold,
                    left_node = self._build_tree(x[left_indices,:],
                                                y[left_indices],
                                                depth + 1),
                    right_node = self._build_tree(x[right_indices,:],
                                                 y[right_indices],
                                                 depth + 1))



