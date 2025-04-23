from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ridge_regression import Regression

reg = Regression()

lambda_list = [0.0001, 0.001, 0.1, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 15000, 20000, 50000]
kfold = 5

X_airline_train = pd.read_csv('Weather/X_train_pca.csv')
X_airline_train = X_airline_train.to_numpy() 
X_airline_train = reg.add_bias_term(X_airline_train)

X_airline_test = pd.read_csv('Weather/X_test_pca.csv')
X_airline_test = X_airline_test.to_numpy() 
X_airline_test = reg.add_bias_term(X_airline_test)

Y_airline_train = pd.read_csv('Weather/y_train.csv')
Y_airline_train = Y_airline_train.to_numpy() 

Y_airline_test = pd.read_csv('Weather/y_test.csv')
Y_airline_test = Y_airline_test.to_numpy() 

best_lambda, best_error, error_list = reg.hyperparameter_search(X_airline_train, Y_airline_train, lambda_list, kfold)

print(best_lambda)
print(best_error)
print(error_list)

weights = reg.ridge_fit_closed(X_airline_train, Y_airline_train, best_lambda)

y_pred = reg.predict(X_airline_test, weights)

rmse = reg.rmse(y_pred,Y_airline_test)

print(rmse)

# PS C:\Users\steve\Documents\CS 7641 Machine Learning\Airline Project> python -u "c:\Users\steve\Documents\CS 7641 Machine Learning\Airline Project\ridge_tests_weather.py"
# 1000
# 49.666518043913406
# [49.666646051736926, 49.66664605157984, 49.66664603430125, 49.66664587726616, 49.66664518026945, 49.66664431117422, 49.66663744432309, 49.666629074976235, 49.666518043913406, 49.6686112356527]
# 47.147622108435755

weights_gd, loss_per_epoch = reg.ridge_fit_GD(
    X_airline_train, Y_airline_train,
    c_lambda=best_lambda,
    epochs=500,          # default
    learning_rate=1e-7   # default
)
# Evaluate on test set
y_pred_gd = reg.predict(X_airline_test, weights_gd)
rmse_gd   = reg.rmse(y_pred_gd, Y_airline_test)
print(f"GD RMSE (test):    {rmse_gd:.4f}")

# PS C:\Users\steve\Documents\CS 7641 Machine Learning\Airline Project> python -u "c:\Users\steve\Documents\CS 7641 Machine Learning\Airline Project\ridge_tests_weather.py"
# 1000
# 0.9942496052900145
# [0.9942521678156959, 0.9942521678125514, 0.9942521674666599, 0.9942521643230517, 0.9942521503702171, 0.9942521329722268, 0.9942519955081105, 0.9942518279662709, 0.9942496052900145, 0.9942915078662882]
# 0.9438250660178839
# GD RMSE (test):    0.9504

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    # plot the y=x reference line
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel('Actual departure delay')
    plt.ylabel('Predicted departure delay')
    plt.title('Closed‑Form Ridge: Actual vs. Predicted')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
plot_actual_vs_predicted(Y_airline_test,y_pred)