import numpy as np

# Define a Node class to represent each node in the Decision Tree
class _Node:
    __slots__ = ("is_leaf", "prediction", "feature_idx", "threshold", "left", "right")
    def __init__(self, is_leaf, prediction=None, feature_idx=None, threshold=None, left=None, right=None):
        self.is_leaf = is_leaf          # Whether this node is a leaf node
        self.prediction = prediction    # Predicted class (0 or 1) for leaf nodes
        self.feature_idx = feature_idx  # Feature index used for splitting
        self.threshold = threshold      # Threshold value for splitting
        self.left = left                # Left child node
        self.right = right              # Right child node

# Define the main Decision Tree class
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, min_impurity_decrease=0.0, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root = None  # Root node of the tree

    # Calculate Gini impurity for a given target array
    @staticmethod
    def _gini(y):
        if y.size == 0:
            return 0.0
        p1 = np.mean(y == 1)
        return 2.0 * p1 * (1.0 - p1)

    # Return the majority class (0 or 1) in the given target array
    def _majority(self, y):
        ones = np.sum(y == 1)
        zeros = y.size - ones
        return 1 if ones >= zeros else 0

    # Find the best feature and threshold to split the dataset
    def _best_split(self, X, y):
        n, d = X.shape
        if n < self.min_samples_split:
            return None, None, 0.0

        parent_imp = self._gini(y)
        best_feat, best_thr, best_gain = None, None, 0.0

        # Try each feature for splitting
        for j in range(d):
            xj = X[:, j]
            uniq = np.unique(xj)
            if uniq.size <= 1:
                continue

            thresholds = (uniq[:-1] + uniq[1:]) / 2.0
            order = np.argsort(xj)
            x_sorted, y_sorted = xj[order], y[order]

            # Evaluate all possible thresholds
            for thr in thresholds:
                idx = np.searchsorted(x_sorted, thr, side="right")
                if idx == 0 or idx == n:
                    continue

                y_left, y_right = y_sorted[:idx], y_sorted[idx:]
                g_left, g_right = self._gini(y_left), self._gini(y_right)
                w_left = y_left.size / n
                impurity = w_left * g_left + (1 - w_left) * g_right
                gain = parent_imp - impurity

                # Keep the best split
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, j, thr

        return best_feat, best_thr, best_gain

    # Recursively build the tree
    def _build(self, X, y, depth):
        # Stop if max depth or not enough samples
        if depth >= self.max_depth or X.shape[0] < self.min_samples_split:
            return _Node(is_leaf=True, prediction=self._majority(y))
        if np.all(y == y[0]):
            return _Node(is_leaf=True, prediction=int(y[0]))

        feat, thr, gain = self._best_split(X, y)
        if feat is None or gain <= self.min_impurity_decrease:
            return _Node(is_leaf=True, prediction=self._majority(y))

        # Split the dataset
        left_mask = X[:, feat] <= thr
        right_mask = ~left_mask
        left_child = self._build(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build(X[right_mask], y[right_mask], depth + 1)

        return _Node(False, None, feat, thr, left_child, right_child)

    # Train the model
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.root = self._build(X, y, depth=0)
        self.root = self._prune(self.root)  # Simplify the tree
        return self

    # Predict the class for a single sample
    def _predict_one(self, x):
        node = self.root
        while not node.is_leaf:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return int(node.prediction)

    # Predict for multiple samples
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row) for row in X], dtype=int)

    # Prune redundant nodes (simplify the tree)
    def _prune(self, node):
        if node.is_leaf:
            return node
        node.left = self._prune(node.left)
        node.right = self._prune(node.right)
        # Merge children if both predict the same value
        if node.left.is_leaf and node.right.is_leaf and node.left.prediction == node.right.prediction:
            return _Node(is_leaf=True, prediction=node.left.prediction)
        return node
