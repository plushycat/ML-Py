import numpy as np
import matplotlib.pyplot as plt

def locally_weighted_regression(x, X, y, tau):
    w = np.exp(-np.sum((X - x)**2, axis=1) / (2 * tau**2)) #gaussian kernel
    W = np.diag(w)
    theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
    return x @ theta

# Data
np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 100)
y = np.sin(X) + 0.1 * np.random.randn(100)
X_bias = np.c_[np.ones(X.shape), X]

# Prediction
x_test = np.linspace(0, 2 * np.pi, 200)
x_test_bias = np.c_[np.ones(x_test.shape), x_test]
tau = 0.5
y_pred = np.array([locally_weighted_regression(xi, X_bias, y, tau) for xi in x_test_bias])

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Training Data', alpha=0.7)
plt.plot(x_test, y_pred, color='blue', label=f'LWR Fit (tau={tau})', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Locally Weighted Regression')
plt.legend()
plt.grid(alpha=0.3)
plt.show()