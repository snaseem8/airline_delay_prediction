import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None
        self.mean = None

    def fit(self, X: np.ndarray) ->None:
        """		
		Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
		You may use the numpy.linalg.svd function
		Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
		corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose
		
		Hint: np.linalg.svd by default returns the transpose of V
		      Make sure you remember to first center your data by subtracting the mean of each feature.
		
		Args:
		    X: (N,D) numpy array corresponding to a dataset
		
		Return:
		    None
		
		Set:
		    self.U: (N, min(N,D)) numpy array
		    self.S: (min(N,D), ) numpy array
		    self.V: (min(N,D), D) numpy array
		"""
        self.mean = np.mean(X, axis=0, keepdims=True)
        centered_X = X - self.mean
        self.U, self.S, self.V = np.linalg.svd(centered_X, full_matrices=False)

    def transform(self, data: np.ndarray, K: int=2) ->np.ndarray:
        """		
		Transform data to reduce the number of features such that final data (X_new) has K features (columns)
		by projecting onto the principal components.
		Utilize class members that were previously set in fit() method.
		
		Args:
		    data: (N,D) numpy array corresponding to a dataset
		    K: int value for number of columns to be kept
		
		Return:
		    X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
        centered_data = data - self.mean
        X_new = centered_data @ self.V[:K, :].T
        return X_new

    def transform_rv(self, data: np.ndarray, retained_variance: float=0.99
        ) ->np.ndarray:
        """		
		Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
		in X_new with K features
		Utilize self.U, self.S and self.V that were set in fit() method.
		
		Args:
		    data: (N,D) numpy array corresponding to a dataset
		    retained_variance: float value for amount of variance to be retained
		
		Return:
		    X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
		           to be kept to ensure retained variance value is retained_variance
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
        total_variance = np.sum(self.S ** 2)
        cumulative_variance = np.cumsum(self.S ** 2) / total_variance
        K = np.argmax(cumulative_variance >= retained_variance) + 1

        centered_data = data - self.mean
        X_new = centered_data @ self.V[:K, :].T
        return X_new

    def get_V(self) ->np.ndarray:
        """		
		Getter function for value of V
		"""
        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig_title) ->None:
        """		
		You have to plot three different scatterplots (2D and 3D for strongest two features and 2D for two random features) for this function.
		For plotting the 2D scatterplots, use your PCA implementation to reduce the dataset to only 2 (strongest and later random) features.
		You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
		Create a scatter plot of the reduced data set and differentiate points that have different true labels using color using matplotlib.
		
		Args:
		    xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
		    ytrain: (N,) numpy array, the true labels
		
		Return: None
		"""
        self.fit(X)
    
        # 2D plot with two strongest components
        X_2d_strong = self.transform(X, K=2)
        
        plt.figure(figsize=(15, 5))
        
        # First subplot: 2D strongest components
        plt.subplot(131)
        scatter = plt.scatter(X_2d_strong[:, 0], X_2d_strong[:, 1], c=y, cmap='viridis')
        plt.title(f'{fig_title}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(scatter)
        
        # 2. 3D plot with three strongest components
        plt.subplot(132, projection='3d')
        X_3d = self.transform(X, K=3)
        scatter = plt.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y, cmap='viridis')
        plt.title(f'{fig_title}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.gca().set_zlabel('PC3')
        plt.colorbar(scatter)
        
        # 3. 2D plot with two random components
        plt.subplot(133)
        n_features = X.shape[1]
        random_idx = np.random.choice(n_features, 2, replace=False)
        X_random = X[:, random_idx]
        scatter = plt.scatter(X_random[:, 0], X_random[:, 1], c=y, cmap='viridis')
        plt.title(f'{fig_title}')
        plt.xlabel(f'Feature {random_idx[0]}')
        plt.ylabel(f'Feature {random_idx[1]}')
        plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.show()
