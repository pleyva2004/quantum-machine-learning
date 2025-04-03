from linear_regression import build_qubo_matrices, mock_qubo_solver, r2_score, adaptive_precision_linreg

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for regression
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate random input features
X = np.random.randn(100, 1)

# Generate target values with some noise
true_slope = 2.5
true_intercept = 1.0
y = true_slope * X + true_intercept + np.random.normal(0, 0.2, (100, 1))

print(f"Generated {len(X)} data points")
print(f"True slope: {true_slope}")
print(f"True intercept: {true_intercept}")

# Plot the data
plt.scatter(X, y, label='Data points')
plt.plot(X, true_slope * X + true_intercept, label='True regression line', color='red')
plt.legend()
plt.show()
# Add bias term to X
X_with_bias = np.hstack([np.ones((len(X), 1)), X])

# Generate precision vector for binary encoding
precision_vector = np.array([1.0, 0.5, 0.25, 0.125])  # More precise binary encoding

# Build QUBO matrices
A, b = build_qubo_matrices(X_with_bias, y.flatten(), precision_vector)


# Get weights from QUBO solution
w_hat = mock_qubo_solver(A, b)

# # Calculate R^2 score
# r2 = r2_score(X_with_bias, y.flatten(), w_hat)

# print(f"Estimated weights: {w_hat}")
# print(f"R^2 score: {r2:.4f}")

# # Plot regression line
# plt.scatter(X, y, label='Data points')
# plt.plot(X, true_slope * X + true_intercept, label='True line', color='red')
# plt.plot(X, X_with_bias @ w_hat, label='QUBO regression', color='green', linestyle='--')
# plt.legend()
# plt.show()










