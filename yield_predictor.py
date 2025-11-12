import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic data (replace with real dataset in production)
num_samples = 1000
temperature = np.random.uniform(15, 35, num_samples)  # degrees C
rainfall = np.random.uniform(500, 2000, num_samples)  # mm/year
pesticides = np.random.uniform(0, 50, num_samples)  # tons

# Synthetic yield formula with noise
yield_true = 10 + 0.5 * rainfall - 0.2 * (temperature - 25)**2 + 0.1 * pesticides + np.random.normal(0, 5, num_samples)

# Normalize features
features = np.column_stack((temperature, rainfall, pesticides))
features_mean = features.mean(axis=0)
features_std = features.std(axis=0)
features_norm = (features - features_mean) / features_std

# Convert to PyTorch tensors
X = torch.tensor(features_norm, dtype=torch.float32)
y = torch.tensor(yield_true, dtype=torch.float32).unsqueeze(1)

# Define the neural network
class YieldPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)  # Input to hidden
        self.fc2 = nn.Linear(64, 32)  # Hidden to hidden
        self.fc3 = nn.Linear(32, 1)   # Hidden to output
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Instantiate model, loss, and optimizer
model = YieldPredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Example prediction
sample_features = np.array([[25, 1000, 20]])  # Temp=25°C, Rain=1000mm, Pest=20 tons
sample_norm = (sample_features - features_mean) / features_std
sample_input = torch.tensor(sample_norm, dtype=torch.float32)
predicted_yield = model(sample_input).item()
print(f'\nPredicted yield: {predicted_yield:.2f} tons/hectare')

# Generate predictions for visualization
with torch.no_grad():
    y_pred = model(X).numpy().flatten()

# Plot predicted vs. actual yields
plt.figure(figsize=(8, 6))
plt.scatter(yield_true, y_pred, alpha=0.5, color='blue')
plt.plot([yield_true.min(), yield_true.max()], [yield_true.min(), yield_true.max()], 'r--', lw=2)
plt.xlabel('Actual Yield (tons/hectare)')
plt.ylabel('Predicted Yield (tons/hectare)')
plt.title('Predicted vs. Actual Crop Yields')
plt.grid(True)
plt.tight_layout()
plt.savefig('yield_plot.png')  # Save for screenshot
plt.show()
