import matplotlib.pyplot as plt

def plot_predicted_vs_actual(y_true, y_pred, save_path="results/predicted_vs_actual.png"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, y_true, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Predicted Delay (minutes)")
    plt.ylabel("Actual Delay (minutes)")
    plt.title("Predicted vs Actual Delays")
    plt.savefig(save_path)
    plt.close()

def plot_residuals(y_true, y_pred, save_path="results/residual_plot.png"):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted Delay (minutes)")
    plt.ylabel("Residuals (minutes)")
    plt.title("Residual Plot")
    plt.savefig(save_path)
    plt.close()

def plot_loss_curve(loss, save_path="results/loss_curve.png", is_sgd=False):
    plt.figure(figsize=(8, 6))
    plt.plot(loss)
    plt.xlabel("Step" if is_sgd else "Epoch")
    plt.ylabel("RMSE Loss")
    plt.title("Loss Curve (SGD)" if is_sgd else "Loss Curve (GD)")
    plt.savefig(save_path)
    plt.close()
