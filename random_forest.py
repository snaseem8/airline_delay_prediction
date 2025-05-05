import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.tree import ExtraTreeClassifier


class RandomForest(object):
    def __init__(self, n_estimators, max_depth, max_features, random_seed=None):
        # helper function. You don't have to modify it
        # Initialization done here
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.max_features = max_features
        self.random_seed = random_seed
        self.bootstraps_row_indices = []
        self.feature_indices = []
        self.out_of_bag = []
        self.decision_trees = [
            ExtraTreeClassifier(max_depth=self.max_depth, criterion="entropy")
            for i in range(self.n_estimators)
        ]
        self.alphas = (
            []
        )  # Importance values for adaptive boosting extra credit implementation

    def _bootstrapping(self, num_training, num_features, random_seed=None):
        """
        TODO:
        - Set random seed if it is inputted
        - Randomly select a sample dataset of size num_training with replacement from the original dataset.
        - Randomly select certain number of features (num_features denotes the total number of features in X,
          max_features denotes the percentage of features that are used to fit each decision tree) without replacement from the total number of features.

        Args:
        - num_training: number of data points in the bootstrapped dataset.
        - num_features: number of features in the original dataset.

        Return:
        - row_idx: the row indices corresponding to the row locations of the selected samples in the original dataset.
        - col_idx: the column indices corresponding to the column locations of the selected features in the original feature list.
        Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
        Hint 1: Please use np.random.choice. First get the row_idx first, and then second get the col_idx.
        Hint 2:  If you are getting a Test Failed: 'bool' object has no attribute 'any' error, please try flooring, or converting to an int, the number of columns needed for col_idx. Using np.ceil() can cause an autograder error.
        """
        if random_seed != None:
            np.random.seed(random_seed)

        row_idx = np.random.choice(num_training, size=num_training, replace=True)
        
        num_selected_features = int(num_features * self.max_features) # convert to an int
        col_idx = np.random.choice(num_features, size=num_selected_features, replace=False) # FALSE!!!

        # print(row_idx)
        # print(col_idx)

        return row_idx, col_idx

    def bootstrapping(self, num_training, num_features):
        # helper function. You don't have to modify it
        # Initializing the bootstap datasets for each tree
        np.random.seed(self.random_seed)
        for i in range(self.n_estimators):
            total = set(list(range(num_training)))
            row_idx, col_idx = self._bootstrapping(num_training, num_features)
            total = total - set(row_idx)
            self.bootstraps_row_indices.append(row_idx)
            self.feature_indices.append(col_idx)
            self.out_of_bag.append(total)

    def fit(self, X, y):
        """
        TODO:
        Train decision trees using the bootstrapped datasets.
        Note that you need to use the row indices and column indices.
        X: NxD numpy array, where N is number
           of instances and D is the dimensionality of each
           instance
        y: 1D numpy array of size (N,), the predicted labels
        Returns:
            None. Calling this function should train the decision trees held in self.decision_trees
        """
        N, D = X.shape

        self.bootstrapping(N,D)

        for i in range(self.n_estimators):

            row_idx = self.bootstraps_row_indices[i]
            col_idx = self.feature_indices[i]

            x_boot = X[row_idx][:,col_idx]
            y_boot = y[row_idx]

            self.decision_trees[i].fit(x_boot, y_boot)
        
        return

    def adaboost(self, X, y):
        """
        TODO:
        - Implement AdaBoost training by adjusting sample weights after each round of training a weak learner.
        - Begin by initializing equal weights for each training sample.
        - For each weak learner:
            - Train the learner on the current sample weights.
            - Get predictions for the training data using the learner's `predict()` method.
            - Calculate the weighted error of the sample and normalize.
            - Calculate `alpha`, the importance of the tree, using the formula from the notebook, and store it.
            - Update the weights of the samples using the formula from the notebook and normalize.

        Args:
        - X: NxD numpy array, feature set.
        - y: 1D numpy array of size (N,), labels.
        Returns:
            None. Trains the ensemble using AdaBoost's weighting mechanism.
        """
        raise NotImplementedError()

    def OOB_score(self, X, y):
        # Helper function. You don't have to modify it.
        # This function computes the accuracy of the random forest model predicting y given x.
        accuracy = []
        for i in range(len(X)):
            predictions = []
            for t in range(self.n_estimators):
                if i in self.out_of_bag[t]:
                    predictions.append(
                        self.decision_trees[t].predict(
                            np.reshape(X[i][self.feature_indices[t]], (1, -1))
                        )[0]
                    )
            if len(predictions) > 0:
                accuracy.append(np.sum(predictions == y[i]) / float(len(predictions)))
        return np.mean(accuracy)

    def predict(self, X):
        N = X.shape[0]
        y = np.zeros((N, 7))
        for t in range(self.n_estimators):
            X_curr = X[:, self.feature_indices[t]]
            y += self.decision_trees[t].predict_proba(X_curr)
        pred = np.argmax(y, axis=1)
        return pred

    def predict_adaboost(self, X):
        # Helper method. You don't have to modify it.
        # This function makes predictions using AdaBoost ensemble by aggregating weighted votes.
        N = X.shape[0]
        weighted_votes = np.zeros((N, 7))

        for alpha, tree in zip(self.alphas, self.decision_trees[: len(self.alphas)]):
            pred = tree.predict(X)
            for i in range(N):
                class_index = int(pred[i])
                weighted_votes[i, class_index] += alpha

        return np.argmax(weighted_votes, axis=1)

    def plot_feature_importance(self, data_train):
        """
        TODO:
        -Display a bar plot showing the feature importance of every feature in
        one decision tree of your choice from the tuned random_forest from Q3.2.
        Args:
            data_train: This is the orginal data train Dataframe containg data AND labels.
                Hint: you can access labels with data_train.columns
        Returns:
            None. Calling this function should simply display the aforementioned feature importance bar chart
        """
        trees = len(self.decision_trees)
        selected_tree = self.decision_trees[np.random.choice(trees)]
        # selected_tree = self.decision_trees[0]

        importances = selected_tree.feature_importances_
        features = data_train.columns[:]

        # print(importances.shape)
        # print(features.shape)

        plt.figure(figsize=(12,6))
        plt.bar(features, importances)
        plt.xlabel("Feature Name")
        plt.ylabel("Importance")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        return
    
    def plot_average_feature_importance(self, data_train):

        total_importances = np.zeros(data_train.shape[1])
    
        for tree, feature_indices in zip(self.decision_trees, self.feature_indices):
            tree_importances = tree.feature_importances_
            total_importances[feature_indices] += tree_importances
        
        avg_importances = total_importances / self.n_estimators
        
        features = data_train.columns.tolist()
        
        plt.figure(figsize=(12, 6))
        plt.bar(features, avg_importances)
        plt.xlabel("Feature Name")
        plt.ylabel("Average Importance")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        return avg_importances

    def hyperparameter_grid_search(
        self, n_estimators_range, max_depth_range, max_features_range
    ):
        """
        Hyperparameter tuning function with customizable ranges.

        Args:
            n_estimators_range (tuple): A tuple (start, stop, step) for n_estimators.
            max_depth_range (tuple): A tuple (start, stop, step) for max_depth.
            max_features_range (tuple): A tuple (start, stop, step) for max_features.

        Note: The range for a hyperparameter is inclusive of start and stop values. i.e.
        (start=8, stop=10, step=1) implies the range of values that that hyperparameter can take on
        is [8, 9, 10].

        Returns:
            A list of tuples, where each tuple represents a unique combination of hyperparameters.

        Example:
            hyperparameter_grid_search((8, 10, 1), (8, 10, 1), (0.7, 1.0, 0.1))

            Output:
            [
                (8, 8, 0.7),
                (8, 8, 0.8),
                ...
                (10, 10, 1.0)
            ]
        """

        nstart, nstop, nstep = n_estimators_range
        dstart, dstop, dstep = max_depth_range
        fstart, fstop, fstep = max_features_range

        # n_estimators_vals = list(range(nstart, nstop+1, nstep))
        n_num = int((nstop-nstart)//nstep)
        n_estimators_vals = np.linspace(nstart, nstop, num=n_num+1,dtype=int)
        # depth_vals = list(range(dstart, dstop+1, dstep))
        d_num = int((dstop-dstart)//dstep)
        depth_vals = np.linspace(dstart, dstop, num=d_num+1, dtype=int)

        f_num = int((fstop-fstart) // fstep)
        feature_vals = np.round(np.linspace(fstart, fstop, num=f_num+1), 10)
        
        hyperparams = []
        for n in n_estimators_vals:
            for d in depth_vals:
                for f in feature_vals:
                    hyperparams.append((n, d, f))

        # print(hyperparams)
        return hyperparams
        
