import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Sample data (replace with your actual data if needed)
X = np.array([660, 700, 710, 720, 740, 780, 820, 840, 860, 880, 900, 920, 940, 960, 1000, 1020, 1040, 1060, 1080, 1100, 1120, 1140, 1180]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 48.19, 47.92, 44.91, 52, 53.11, 53.022, 55.044, 56.11, 55.556, 57, 58.34, 58.867, 58.122, 59.389, 59.61, 62.4, 61.76, 63.3]).reshape(-1, 1)

# Create the polynomial features
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Fit the linear regression model
reg = LinearRegression()
reg.fit(X_poly, y)

# Generate data points for the fitted line (optional, adjust range as needed)
X_values = np.linspace(min(X), max(X), 100).reshape(-1, 1)  # Create 100 data points within the data range
X_values_poly = poly_features.fit_transform(X_values)
y_values = reg.predict(X_values_poly)

# Create the plot
plt.figure(figsize=(10, 6))  # Adjust figure size for better visualization

# Scatter plot
plt.scatter(X, y, color='blue', label='Data Points')

# Line plot (ensure it doesn't intersect the x-axis at origin)
plt.plot(X_values, y_values, color='red', label='Fitted Line')

# Customize the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.title('Radiation count vs. applied voltage')
plt.grid(True)  # Add grid lines for better readability
plt.legend()  # Add legend

# Set x-axis limits to explicitly exclude the origin
plt.xlim(min(X) - 20, max(X) + 20)  # Adjust the range as needed

# Display the plot
plt.show()