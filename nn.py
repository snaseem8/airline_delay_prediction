import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load PCA-transformed features
X_train_pca = pd.read_csv("X_train_pca.csv").to_numpy()
X_test_pca = pd.read_csv("X_test_pca.csv").to_numpy()

# Load target values
y_train = pd.read_csv("y_train.csv").to_numpy()
y_test = pd.read_csv("y_test.csv").to_numpy()

# Convert features and targets to PyTorch tensors
X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define a simple feedforward neural network for regression
class DelayPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# Initialize model, loss function, and optimizer
input_dim = X_train_tensor.shape[1]
model = DelayPredictor(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-3)

# Training loop
epochs = 3000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_mse = mean_squared_error(y_test_tensor.numpy(), test_preds.numpy())
    print(f"\nTest MSE: {test_mse:.4f}")
