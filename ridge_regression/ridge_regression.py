
from typing import List, Tuple
import numpy as np


class Regression(object):

    def __init__(self):
        pass

    def add_bias_term(self, x: np.ndarray) -> np.ndarray:
        """
        Given feature data x, prepend a bias column of ones.

        Args:
            x: 
                1-D array of shape (N,)   or
                2-D array of shape (N, D)

        Returns:
            X_bias: 2-D array of shape (N, D+1), where
                X_bias[:, 0] == 1 for all rows.
        """
        # ensure we have a 2-D array of shape (N, D)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        N = x.shape[0]
        ones = np.ones((N, 1), dtype=x.dtype)

        # stack the bias column on the left
        X_bias = np.hstack([ones, x])
        return X_bias    
    
    
    def rmse(self, pred: np.ndarray, label: np.ndarray) ->float:
        """		
        Calculate the root mean square error.
        
        Args:
            pred: (N, 1) numpy array, the predicted labels
            label: (N, 1) numpy array, the ground truth labels
        Return:
            A float value
        """

        N = pred.shape[0]

        diff = pred - label
        
        total = np.sum(diff ** 2)
        
        rmse = np.sqrt(total / N)
        
        return float(rmse)

      

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
        
        prediction = np.sum(xtest * weight.T, axis = 1, keepdims=True)
        
        return prediction     
    

    def ridge_fit_closed(self, xtrain: np.ndarray, ytrain: np.ndarray,
        c_lambda: float) ->np.ndarray:
        """		
        Fit a ridge regression model using the closed form solution
        
        Args:
            xtrain: (N,1+D) numpy array, where N is
                    number of instances and D is the dimensionality
                    of each instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
        Return:
            weight: (1+D,1) numpy array, the weights of ridge regression model
        Hints:
            - You should adjust your I matrix to handle the bias term differently than the rest of the terms
        """
        
        trans = xtrain.T @ xtrain
        I = np.eye(xtrain.shape[1])
        I[0,0] = 0
        parent = np.linalg.inv(trans + c_lambda * I)
        weight = parent @ xtrain.T @ ytrain
        
        return weight
    
    def ridge_fit_GD(self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda:
        float, epochs: int=500, learning_rate: float=1e-07) ->Tuple[np.
        ndarray, List[float]]:
        """		
        Fit a ridge regression model using gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.
        
        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        Hints:
            - RMSE loss should be recorded AFTER the gradient update in each iteration.
            - You should avoid applying regularization to the bias term in the gradient update
        """
        
        N = xtrain.shape[0]
        D = xtrain.shape[1] - 1
        
        weight = np.zeros([D+1,1])
        loss_per_epoch = []
        
        for _ in range(epochs):
            ypredict = xtrain @ weight
            reg_term = c_lambda * weight
            reg_term[0] = 0
            weight = weight - (learning_rate / N) * (xtrain.T @ (ypredict - ytrain) + reg_term)
            ypredict_updated = xtrain @ weight
            rmse_val = self.rmse(ypredict_updated, ytrain)
            loss_per_epoch.append(rmse_val)
        
        return weight, loss_per_epoch
    

    def ridge_fit_SGD(self, xtrain: np.ndarray, ytrain: np.ndarray,
        c_lambda: float, epochs: int=100, learning_rate: float=0.001) ->Tuple[
        np.ndarray, List[float]]:
        """		
        Fit a ridge regression model using stochastic gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.
        
        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float, value of regularization constant
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
            - You should avoid applying regularization to the bias term in the gradient update
        
        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.
        """
        
        N = xtrain.shape[0]
        D = xtrain.shape[1] - 1
        
        weight = np.zeros([D+1, 1])
        loss_per_step = []
        
        for _ in range(epochs):
            for i in range(N):
                ypredict = xtrain[i,:] @ weight
                reg_term = c_lambda * weight
                reg_term[0] = 0
                weight = weight - learning_rate * (((xtrain[i, :].reshape(-1, 1)) * (ypredict - ytrain[i])) + reg_term / N)
                ypredict_updated = xtrain @ weight
                rmse_val = self.rmse(ypredict_updated, ytrain)
                loss_per_step.append(rmse_val)
        
        return weight, loss_per_step
    

    def ridge_cross_validation(self, X: np.ndarray, y: np.ndarray, kfold:
        int=5, c_lambda: float=100) ->List[float]:
        """		
        For each of the k-folds of the provided X, y data, fit a ridge regression model
        and then evaluate the RMSE. Return the RMSE for each fold
        
        Args:
            X : (N,1+D) numpy array, where N is the number of instances
                and D is the dimensionality of each instance with a bias term
            y : (N,1) numpy array, true labels
            kfold: int, number of folds you should take while implementing cross validation.
            c_lambda: float, value of regularization constant
        Returns:
            loss_per_fold: list[float], RMSE loss for each kfold
        Hints:
            - np.concatenate might be helpful.
            - Use ridge_fit_closed for this function.
            - Look at 3.5 to see how this function is being used.
            - For kfold=5:
                split X and y into 5 equal-size folds
                use 80 percent for training and 20 percent for test
        """
        
        N = X.shape[0]
        D = X.shape[1] - 1
        length = N / kfold
        loss_per_fold = []
        
        splits_x = np.array_split(X, kfold)
        splits_y = np.array_split(y, kfold)
        
        for i in range(kfold):
            X_test = splits_x[i]
            y_test = splits_y[i]
            
            X_train = np.concatenate([splits_x[j] for j in range(kfold) if j != i], axis=0)
            y_train = np.concatenate([splits_y[j] for j in range(kfold) if j != i], axis=0)
            
            weight = self.ridge_fit_closed(X_train, y_train, c_lambda)
            
            y_predicted = X_test @ weight
            
            rmse_val = self.rmse(y_predicted, y_test)
            loss_per_fold.append(rmse_val)
        
        return loss_per_fold
    

    def hyperparameter_search(self, X: np.ndarray, y: np.ndarray,
        lambda_list: List[float], kfold: int) ->Tuple[float, float, List[float]
        ]:
        """
        FUNCTION PROVIDED TO STUDENTS

        Search over the given list of possible lambda values lambda_list
        for the one that gives the minimum average error from cross-validation

        Args:
            X : (N, 1+D) numpy array, where N is the number of instances and
                D is the dimensionality of each instance with a bias term
            y : (N,1) numpy array, true labels
            lambda_list: list of regularization constants (lambdas) to search from
            kfold: int, Number of folds you should take while implementing cross validation.
        Returns:
            best_lambda: (float) the best value for the regularization const giving the least RMSE error
            best_error: (float) the average RMSE error achieved using the best_lambda
            error_list: list[float] list of average RMSE loss for each lambda value given in lambda_list
        """
        best_error = None
        best_lambda = None
        error_list = []
        for lm in lambda_list:
            err = self.ridge_cross_validation(X, y, kfold, lm)
            mean_err = np.mean(err)
            error_list.append(mean_err)
            if best_error is None or mean_err < best_error:
                best_error = mean_err
                best_lambda = lm
        return best_lambda, best_error, error_list  
    
    
    # def construct_polynomial_feats(self, x: np.ndarray, degree: int
    #     ) ->np.ndarray:
    #     """		
    #     Given a feature matrix x, create a new feature matrix
    #     which is all the possible combinations of polynomials of the features
    #     up to the provided degree
        
    #     Args:
    #         x:
    #             1-dimensional case: (N,) numpy array
    #             D-dimensional case: (N, D) numpy array
    #             Here, N is the number of instances and D is the dimensionality of each instance.
    #         degree: the max polynomial degree
    #     Return:
    #         feat:
    #             For 1-D array, numpy array of shape Nx(degree+1), remember to include
    #             the bias term. feat is in the format of:
    #             [[1.0, x1, x1^2, x1^3, ....,],
    #              [1.0, x2, x2^2, x2^3, ....,],
    #              ......
    #             ]
    #     Hints:
    #         - For D-dimensional array: numpy array of shape N x (degree+1) x D, remember to include
    #         the bias term.
    #         - It is acceptable to loop over the degrees.
    #         - Example:
    #         For inputs x: (N = 3 x D = 2) and degree: 3,
    #         feat should be:
        
    #         [[[ 1.0        1.0]
    #             [ x_{1,1}    x_{1,2}]
    #             [ x_{1,1}^2  x_{1,2}^2]
    #             [ x_{1,1}^3  x_{1,2}^3]]
        
    #             [[ 1.0        1.0]
    #             [ x_{2,1}    x_{2,2}]
    #             [ x_{2,1}^2  x_{2,2}^2]
    #             [ x_{2,1}^3  x_{2,2}^3]]
        
    #             [[ 1.0        1.0]
    #             [ x_{3,1}    x_{3,2}]
    #             [ x_{3,1}^2  x_{3,2}^2]
    #             [ x_{3,1}^3  x_{3,2}^3]]]
    #     """
        
    #     original_ndim = x.ndim
    #     N = x.shape[0]
       
    #     if len(x.shape) == 1:
    #         x = x.reshape(-1, 1)
    #     D = x.shape[1]
        
    #     feat = []
            
    #     for i in range(N):
    #         arr = []
    #         arr.append(np.ones(D))
    #         for j in range(1,degree+1):
    #             arr.append(x[i,:] ** j)
                
    #         feat.append(arr)
            
    #     if original_ndim == 1:
    #         feat = feat.squeeze(-1)
            
    #     return np.array(feat)