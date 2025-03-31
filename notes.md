# Roadmap for Continuous Linear Regression Model

## 1. Data Preprocessing
- **Objective**: Prepare the data for a continuous regression task.
- **Steps**:
  - Merge flight and weather datasets using a common key (e.g., date and airport).
  - Define the target variable (`y`) as delay time in minutes (continuous).
  - Select relevant features (`X`) from both datasets (e.g., departure hour, temperature, wind speed, precipitation).
  - Handle missing values (e.g., impute with mean or drop rows).
  - Convert categorical variables (e.g., airline) to numerical format (e.g., one-hot encoding).
  - Normalize numerical features (e.g., weather variables) to a consistent scale (e.g., 0 to 1).
  - Add a bias term (column of ones) to `X` to match the expected input format of your class (shape: `(N, 1+D)`).
  - Split the data into training (e.g., 80%) and testing (e.g., 20%) sets using a random split.

## 2. Model Training
- **Objective**: Train the linear regression model using your class methods.
- **Steps**:
  - **Closed-Form Solution**:
    - Use `linear_fit_closed` to compute weights directly on the training data (`xtrain`, `ytrain`).
    - Provides a quick baseline with no iteration required.
  - **Gradient Descent (GD)**:
    - Use `linear_fit_GD` with reasonable defaults (e.g., `epochs=100`, `learning_rate=0.001`).
    - Record the `loss_per_epoch` to monitor convergence.
  - **Stochastic Gradient Descent (SGD)**:
    - Use `linear_fit_SGD` with similar defaults (e.g., `epochs=10`, `learning_rate=0.001`).
    - Record the `loss_per_step` for finer-grained analysis.
  - **Polynomial Features (Optional)**:
    - Use `construct_polynomial_feats` with a small degree (e.g., 2) to test if non-linear terms improve performance.
    - Retrain the model with the expanded feature set.
  - Choose the best method (closed-form, GD, SGD, or polynomial) based on training RMSE or computational efficiency.

## 3. Model Testing
- **Objective**: Evaluate the model’s performance on the test set.
- **Steps**:
  - Use `predict` with the trained weights and test features (`xtest`) to generate predictions (`pred`).
  - Compute quantitative metrics on the test set:
    - **Root Mean Squared Error (RMSE)**: Use `rmse` to measure prediction error in minutes.
  - Compare training RMSE (from GD/SGD or closed-form) to test RMSE to check for overfitting.

## 4. Quantitative Results
- **Objective**: Summarize model performance numerically.
- **Steps**:
  - Calculate RMSE for the test set predictions.
  - Optionally, compute these metrics for the training set to assess fit.
  - Record training time for each method (closed-form, GD, SGD) using Python’s `time` module.
  - Present results in a table format (e.g., using pandas DataFrame) with columns:
    - Method (Closed-Form, GD, SGD, Polynomial)
    - Training RMSE
    - Test RMSE
    - Training Time (seconds)

## 5. Visual Results
- **Objective**: Create visualizations to interpret model performance and data relationships.
- **Steps**:
  - **Predicted vs. Actual Plot**:
    - Scatter plot of predicted delays (`pred`) vs. actual delays (`ytest`).
    - Add a 45-degree line (`y=x`) to show perfect predictions.
    - Use different colors or transparency to highlight density.
  - **Residual Plot**:
    - Scatter plot of residuals (`ytest - pred`) vs. predicted values (`pred`).
    - Look for patterns (e.g., spread increasing with predictions indicates heteroscedasticity).
    - Add a horizontal line at y=0 for reference.
  - **Loss Convergence (GD/SGD)**:
    - Plot `loss_per_epoch` (from GD) vs. epoch number to show training convergence.
    - Plot `loss_per_step` (from SGD) vs. step number (or subset for readability).
  - **Feature Importance**:
    - Bar chart of the weights from the closed-form solution to show which features (e.g., wind speed, temperature) have the largest impact on delays.
  - **Data Exploration**:
    - Scatter plot of key features (e.g., wind speed, precipitation) vs. delay time to visualize relationships.

## 6. Final Output
- **Objective**: Compile and interpret results for presentation.
- **Steps**:
  - Save quantitative results as a table (e.g., CSV or printed output).
  - Save visualizations as image files (e.g., PNG) using matplotlib’s `savefig`.
  - Interpret findings:
    - How well does the model predict delays (based on RMSE)?
    - Do GD and SGD converge to similar results as the closed-form solution?
    - Which features are most predictive of delays?
  - Suggest next steps (e.g., adding more features, trying polynomial terms).

## Tools and Libraries
- **Pandas**: For data loading, merging, and preprocessing.
- **NumPy**: For matrix operations (already in your class).
- **Matplotlib/Seaborn**: For plotting.
- **Scikit-learn**: For:
  - Train-test split (`train_test_split`).
  - Metrics (`mean_absolute_error`, `r2_score`).
  - Optional: Validation with `sklearn.linear_model.LinearRegression`.
- **Time**: To measure training duration.

## Example Workflow
1. **Preprocess**: Load data, merge, clean, split into `xtrain`, `xtest`, `ytrain`, `ytest`.
2. **Train**: Fit model with `linear_fit_closed` and `linear_fit_GD`.
3. **Test**: Predict on `xtest`, compute RMSE.
4. **Visualize**: Generate predicted vs. actual, residual, and loss plots.
5. **Summarize**: Compile metrics into a table and save plots.


## Folder Structure
flight_delay_prediction/
├── regression.py        # Your Regression class
├── plotting.py         # Plotting functions
├── main.ipynb          # Jupyter Notebook to run everything
├── data/               # Folder for your datasets (e.g., flights.csv, weather.csv)
├── results/            # Folder to save plots and quantitative results
│   ├── predicted_vs_actual.png
│   ├── residual_plot.png
│   ├── loss_curve.png
│   └── metrics.csv
└── README.md           # Overview for GitHub Pages


### Example helper function for preparing features
import numpy as np

class Regression:
    # Your existing code here (rmse, construct_polynomial_feats, predict, etc.)

def prepare_features(X, degree=None):
    """Helper to prepare feature matrix with optional polynomial expansion."""
    reg = Regression()
    if degree is not None:
        X_poly = reg.construct_polynomial_feats(X, degree)
        if X.ndim > 1:  # 2D case
            N = X.shape[0]
            X_poly = X_poly.reshape(N, -1)
            # Keep only one bias column
            X_poly = np.hstack([X_poly[:, :1], X_poly[:, degree+1:]])
        return X_poly
    else:
        # Add bias term manually if no polynomial
        return np.hstack([np.ones((X.shape[0], 1)), X])

## Main.ipynb structure
### Setup
import pandas as pd
import numpy as np
from regression import Regression, prepare_features
from plotting import plot_predicted_vs_actual, plot_residuals, plot_loss_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

### Load and preprocess data
flights = pd.read_csv("data/flights.csv")
weather = pd.read_csv("data/weather.csv")
data = pd.merge(flights, weather, on=["date", "airport"])
X = data[["wind_speed", "temperature", "precipitation"]].values
y = data["delay_minutes"].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_poly = prepare_features(X_train, degree=2)
X_test_poly = prepare_features(X_test, degree=2)

## Randomly sample subset of data
sample_percentage = 0.5  # 50%
total_rows = X.shape[0]
sample_size = int(total_rows * sample_percentage)  # Number of rows to sample

indices = np.random.choice(total_rows, size=sample_size, replace=False)

X_sampled = X[indices]
y_sampled = y[indices]

print(f"Original shape: X={X.shape}, y={y.shape}")
print(f"Sampled shape: X={X_sampled.shape}, y={y_sampled.shape}")

### Train Model
reg = Regression()

w_closed = reg.linear_fit_closed(X_train_poly, y_train)
y_pred_closed = reg.predict(X_test_poly, w_closed)
rmse_closed = reg.rmse(y_pred_closed, y_test)

w_gd, loss_gd = reg.linear_fit_GD(X_train_poly, y_train, epochs=100, learning_rate=0.001)
y_pred_gd = reg.predict(X_test_poly, w_gd)
rmse_gd = reg.rmse(y_pred_gd, y_test)

### Visualizations
plot_predicted_vs_actual(y_test, y_pred_closed)
plot_residuals(y_test, y_pred_closed)
plot_loss_curve(loss_gd)
