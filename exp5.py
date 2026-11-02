
# Assignment 5: Ridge and Lasso Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
data = load_diabetes(as_frame=True)
X = data.data
y = data.target

print("Dataset Shape:", X.shape)

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Create models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=10000)
}

results = []

# 5. Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    train_rmse = mean_squared_error(y_train, train_pred) ** 0.5
    test_rmse = mean_squared_error(y_test, test_pred) ** 0.5

    results.append([name, train_r2, test_r2, train_rmse, test_rmse])

    print("\n", name)
    print("Train R2 Score:", round(train_r2, 3))
    print("Test R2 Score:", round(test_r2, 3))
    print("Train RMSE:", round(train_rmse, 3))
    print("Test RMSE:", round(test_rmse, 3))

# 6. Display comparison
result_df = pd.DataFrame(results, columns=[
    "Model", "Train R2", "Test R2", "Train RMSE", "Test RMSE"
])

print("\nModel Comparison:")
print(result_df.to_string(index=False))

# 7. Plot test R2 scores
plt.bar(result_df["Model"], result_df["Test R2"])
plt.xlabel("Regression Models")
plt.ylabel("Test R2 Score")
plt.title("Comparison of Regression Models")
plt.tight_layout()
plt.show()