
from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ridge_regression import Regression

reg = Regression()

lambda_list = [0.0001, 0.001, 0.1, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 15000, 20000, 50000]
kfold = 5

X_airline_train = pd.read_csv('Both/X_train_pca.csv')
X_airline_train = X_airline_train.to_numpy() 
X_airline_train = reg.add_bias_term(X_airline_train)

X_airline_test = pd.read_csv('Both/X_test_pca.csv')
X_airline_test = X_airline_test.to_numpy() 
X_airline_test = reg.add_bias_term(X_airline_test)

Y_airline_train = pd.read_csv('Both/y_train.csv')
Y_airline_train = Y_airline_train.to_numpy() 

Y_airline_test = pd.read_csv('Both/y_test.csv')
Y_airline_test = Y_airline_test.to_numpy() 

best_lambda, best_error, error_list = reg.hyperparameter_search(X_airline_train, Y_airline_train, lambda_list, kfold)

#best_lambda = 0

print(best_lambda)
print(best_error)
print(error_list)

weights = reg.ridge_fit_closed(X_airline_train, Y_airline_train, best_lambda)

y_pred = reg.predict(X_airline_test, weights)

rmse = reg.rmse(y_pred,Y_airline_test)

print(rmse)

############################################################################################################################################

############################################################################################################################################


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

############################################################################################################################################

############################################################################################################################################

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 50):
    res = (y_true - y_pred).flatten()
    plt.figure(figsize=(6,4))
    plt.hist(res, bins=bins, edgecolor='k', alpha=0.7)
    plt.xlabel('Residual (Actual – Predicted)')
    plt.ylabel('Count')
    plt.title('Residual Distribution')
    plt.tight_layout()
    plt.show()

plot_residuals(Y_airline_test, y_pred)

############################################################################################################################################

############################################################################################################################################

weights_gd, loss_per_epoch = reg.ridge_fit_GD(
    X_airline_train, Y_airline_train,
    c_lambda=best_lambda,
    epochs=3000,          # default
    learning_rate=1e-3 
)

# Evaluate on test set
y_pred_gd = reg.predict(X_airline_test, weights_gd)
rmse_gd   = reg.rmse(y_pred_gd, Y_airline_test)
print(f"GD RMSE (test):    {rmse_gd:.4f}")

def plot_learning_curve(losses: list, title: str = 'Learning Curve'):
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(losses)+1), losses, lw=2)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_learning_curve(loss_per_epoch, 
    title=f'Ridge GD (λ={best_lambda}) — RMSE per Epoch')


# y_pred_minutes = scaler_y.inverse_transform(y_pred)
# y_test_minutes = scaler_y.inverse_transform(Y_airline_test)
# rmse_minutes = reg.rmse(y_pred_minutes, y_test_minutes)

#print(rmse_minutes)

# PS C:\Users\steve\Documents\CS 7641 Machine Learning\Airline Project> python -u "c:\Users\steve\Documents\CS 7641 Machine Learning\Airline Project\ridge_test_both.py"
# 50000
# 0.9872797129575945
# [1.0628055857760785, 1.062805395180941, 1.0627844346307218, 1.062594329929068, 1.0617590241698431, 1.0607364399970978, 1.053329448812254, 1.0456614482925457, 1.014966220065224, 1.002517801975374, 0.9906052093870421, 0.9891386659233362, 0.9884668813628054, 0.9880359204221486, 0.9872797129575945, 0.9880564836193706, 0.9892051404177902, 0.9902657055079483]
# 0.9359988302613238
# GD RMSE (test):    0.9504


lams = np.log10(lambda_list)
plt.plot(lams, error_list, '-o')
plt.xlabel('log10(lambda)')
plt.ylabel('CV RMSE')
plt.title('Ridge CV curve')
plt.show()

# 2) re‐train at a smaller best_lambda (if it came out huge):
best_lambda = max(best_lambda, 1e-2)  # floor it
w = reg.ridge_fit_closed(X_airline_train, Y_airline_train, best_lambda)

# 3) scatter residuals vs actual
y_pred = reg.predict(X_airline_test, w)
plt.scatter(Y_airline_test, Y_airline_test - y_pred, alpha=0.3, s=10)
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Actual delay')
plt.ylabel('Residual')
plt.title('Residual vs Actual')
plt.show()