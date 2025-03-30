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
