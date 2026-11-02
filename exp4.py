# Assignment 4: Polynomial Regression with Different Degrees

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
data = fetch_california_housing()
X = data.data[:, [0]]  # Median Income
y = data.target
'''
from sklearn.datasets import load_diabetes
# 1. Load built-in dataset (works offline)
data = load_diabetes()

# Use BMI feature for polynomial regression
X = data.data[:, [2]]
y = data.target

'''
# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Compare different polynomial degrees
degrees = [1, 2, 3, 5, 10]

train_errors = []
test_errors = []

for degree in degrees:
    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        StandardScaler(),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    train_errors.append(train_mse)
    test_errors.append(test_mse)

    print("\nPolynomial Degree:", degree)
    print("Training MSE:", round(train_mse, 4))
    print("Testing MSE:", round(test_mse, 4))
    print("Testing R2 Score:", round(r2_score(y_test, test_pred), 4))

# 4. Plot training and testing errors
plt.plot(degrees, train_errors, marker='o', label="Training MSE")
plt.plot(degrees, test_errors, marker='s', label="Testing MSE")

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Polynomial Regression: Bias-Variance Trade-off")
plt.legend()
plt.grid(True)
plt.show()