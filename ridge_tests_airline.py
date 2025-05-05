from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ridge_regression import Regression

reg = Regression()

lambda_list = [0.0001, 0.001, 0.1, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 15000, 20000, 50000]
kfold = 5

X_airline_train = pd.read_csv('Airline/X_train_pca_airK.csv')
X_airline_train = X_airline_train.to_numpy() 
X_airline_train = reg.add_bias_term(X_airline_train)

X_airline_test = pd.read_csv('Airline/X_test_pca_airK.csv')
X_airline_test = X_airline_test.to_numpy() 
X_airline_test = reg.add_bias_term(X_airline_test)

Y_airline_train = pd.read_csv('Airline/y_train_airK.csv')
Y_airline_train = Y_airline_train.to_numpy() 

Y_airline_test = pd.read_csv('Airline/y_test_airK.csv')
Y_airline_test = Y_airline_test.to_numpy() 

best_lambda, best_error, error_list = reg.hyperparameter_search(X_airline_train, Y_airline_train, lambda_list, kfold)

print(best_lambda)
print(best_error)
print(error_list)

weights = reg.ridge_fit_closed(X_airline_train, Y_airline_train, best_lambda)

y_pred = reg.predict(X_airline_test, weights)

rmse = reg.rmse(y_pred,Y_airline_test)

print(rmse)

# PS C:\Users\steve\Documents\CS 7641 Machine Learning\Airline Project> python -u "c:\Users\steve\Documents\CS 7641 Machine Learning\Airline Project\ridge_tests.py"
# 10000
# 49.57900142011831
# [49.956726077536274, 49.956723695936844, 49.95646207121196, 49.95411510567812, 49.94432020828414, 49.933342600159364, 49.87455904771291, 49.83269154396487, 49.68081517448963, 49.57900142011831]
# 47.01126683366134

weights_gd, loss_per_epoch = reg.ridge_fit_GD(
    X_airline_train, Y_airline_train,
    c_lambda=best_lambda,
    epochs=10000,          # default
    learning_rate=1e-3   # default
)
# Evaluate on test set
y_pred_gd = reg.predict(X_airline_test, weights_gd)
rmse_gd   = reg.rmse(y_pred_gd, Y_airline_test)
print(f"GD RMSE (test):    {rmse_gd:.4f}")


# PS C:\Users\steve\Documents\CS 7641 Machine Learning\Airline Project> python -u "c:\Users\steve\Documents\CS 7641 Machine Learning\Airline Project\ridge_tests.py"
# 10000
# 0.9924976530273723
# [1.0000591372291452, 1.000059089553077, 1.000053852216342, 1.0000068694672475, 0.9998107902321539, 0.9995910348924413, 0.9984142758579997, 0.9975761508853738, 0.9945358125175799, 0.9924976530273723]
# 0.9410954368136822
# GD RMSE (test):    0.9504