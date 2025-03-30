from typing import List, Tuple
import numpy as np


class LinearRegression(object):

    def __init__(self):
        pass

    def rmse(self, pred: np.ndarray, label: np.ndarray) ->float:
        """		
		Calculate the root mean square error.
		
		Args:
		    pred: (N, 1) numpy array, the predicted labels
		    label: (N, 1) numpy array, the ground truth labels
		Return:
		    A float value
		"""
        return np.sqrt(np.sum((label - pred) ** 2) / pred.shape[0])

    def construct_polynomial_feats(self, x: np.ndarray, degree: int
        ) ->np.ndarray:
        """		
		Given a feature matrix x, create a new feature matrix
		which is all the possible combinations of polynomials of the features
		up to the provided degree
		
		Args:
		    x:
		        1-dimensional case: (N,) numpy array
		        D-dimensional case: (N, D) numpy array
		        Here, N is the number of instances and D is the dimensionality of each instance.
		    degree: the max polynomial degree
		Return:
		    feat:
		        For 1-D array, numpy array of shape Nx(degree+1), remember to include
		        the bias term. feat is in the format of:
		        [[1.0, x1, x1^2, x1^3, ....,],
		         [1.0, x2, x2^2, x2^3, ....,],
		         ......
		        ]
		Hints:
		    - For D-dimensional array: numpy array of shape N x (degree+1) x D, remember to include
		    the bias term.
		    - It is acceptable to loop over the degrees.
		    - Example:
		    For inputs x: (N = 3 x D = 2) and degree: 3,
		    feat should be:
		
		    [[[ 1.0        1.0]
		        [ x_{1,1}    x_{1,2}]
		        [ x_{1,1}^2  x_{1,2}^2]
		        [ x_{1,1}^3  x_{1,2}^3]]
		
		        [[ 1.0        1.0]
		        [ x_{2,1}    x_{2,2}]
		        [ x_{2,1}^2  x_{2,2}^2]
		        [ x_{2,1}^3  x_{2,2}^3]]
		
		        [[ 1.0        1.0]
		        [ x_{3,1}    x_{3,2}]
		        [ x_{3,1}^2  x_{3,2}^2]
		        [ x_{3,1}^3  x_{3,2}^3]]]
		"""
        # check input size
        size = x.ndim
        
        # first case
        if size == 1:
            feat = np.ones((x.shape[0], degree + 1))
            for deg in np.arange(degree):
                feat[:, deg+1] = x ** (deg+1)
        
        # second case
        else:
            feat = np.ones((x.shape[0], degree + 1, x.shape[1]))
            for deg in np.arange(degree):
                feat[:, deg+1, :] = x ** (deg+1)
                
        return feat

    def predict(self, xtest: np.ndarray, weight: np.ndarray) ->np.ndarray:
        """		
		Using regression weights, predict the values for each data point in the xtest array
		
		Args:
		    xtest: (N,1+D) numpy array, where N is the number
		            of instances and D is the dimensionality
		            of each instance with a bias term
		    weight: (1+D,1) numpy array, the weights of linear regression model
		Return:
		    prediction: (N,1) numpy array, the predicted labels
		"""
        return np.dot(xtest, weight)

    def linear_fit_closed(self, xtrain: np.ndarray, ytrain: np.ndarray
        ) ->np.ndarray:
        """		
		Fit a linear regression model using the closed form solution
		
		Args:
		    xtrain: (N,1+D) numpy array, where N is number
		            of instances and D is the dimensionality
		            of each instance with a bias term
		    ytrain: (N,1) numpy array, the true labels
		Return:
		    weight: (1+D,1) numpy array, the weights of linear regression model
		Hints:
		    - For pseudo inverse, you should use the numpy linear algebra function (np.linalg.pinv)
		"""
        return np.linalg.pinv(xtrain.T @ xtrain) @ (xtrain.T @ ytrain)

    def linear_fit_GD(self, xtrain: np.ndarray, ytrain: np.ndarray, epochs:
        int=5, learning_rate: float=0.001) ->Tuple[np.ndarray, List[float]]:
        """		
		Fit a linear regression model using gradient descent.
		Although there are many valid initializations, to pass the local tests
		initialize the weights with zeros.
		
		Args:
		    xtrain: (N,1+D) numpy array, where N is number
		            of instances and D is the dimensionality
		            of each instance with a bias term
		    ytrain: (N,1) numpy array, the true labels
		Return:
		    weight: (1+D,1) numpy array, the weights of linear regression model
		    loss_per_epoch: (epochs,) list of floats, rmse of each epoch
		Hints:
		    - RMSE loss should be recorded AFTER the gradient update in each iteration.
		""" 
        weights = np.zeros((xtrain.shape[1], 1))
        loss_per_epoch = []
        N = xtrain.shape[0]
        
        for epoc in range(epochs):
            y_predicted = xtrain @ weights
            weights = weights + (learning_rate / N) * xtrain.T @ (ytrain - y_predicted)
            y_predicted_updated = xtrain @ weights
            rmse = self.rmse(y_predicted_updated, ytrain)
            loss_per_epoch.append(float(rmse))  
            
        return weights, loss_per_epoch

    def linear_fit_SGD(self, xtrain: np.ndarray, ytrain: np.ndarray, epochs:
        int=100, learning_rate: float=0.001) ->Tuple[np.ndarray, List[float]]:
        """		
		Fit a linear regression model using stochastic gradient descent.
		Although there are many valid initializations, to pass the local tests
		initialize the weights with zeros.
		
		Args:
		    xtrain: (N,1+D) numpy array, where N is number
		            of instances and D is the dimensionality of each
		            instance with a bias term
		    ytrain: (N,1) numpy array, the true labels
		    epochs: int, number of epochs
		    learning_rate: float
		Return:
		    weight: (1+D,1) numpy array, the weights of linear regression model
		    loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step
		Hints:
		    - RMSE loss should be recorded AFTER the gradient update in each iteration.
		    - Keep in mind that the number of epochs is the number of complete passes
		    through the training dataset. SGD updates the weight for one datapoint at
		    a time. For each epoch, you'll need to go through all of the points.
		
		NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.
		"""
        weights = np.zeros((xtrain.shape[1], 1))
        loss_per_epoch = []
        N = xtrain.shape[0]
        
        for epoc in range(epochs):
            for i in range(N):
                y_predicted = xtrain @ weights
                weights = weights + learning_rate * xtrain[i:i+1].T @ (ytrain[i:i+1] - y_predicted[i:i+1])
                y_predicted_updated = xtrain @ weights
                rmse = self.rmse(y_predicted_updated, ytrain)
                loss_per_epoch.append(float(rmse))  
            
        return weights, loss_per_epoch
