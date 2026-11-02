import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the Wine dataset and convert to a binary classification problem
wine = load_wine()
X = wine.data
# Target 0 vs non-0 (binary classification)
y = (wine.target == 0).astype(int)
target_names = ["Not Class 0", "Class 0"]

# 2. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)

# 3. Standardize features for optimal convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Initialize and train Logistic Regression model
lr_model = LogisticRegression(random_state=100)
lr_model.fit(X_train_scaled, y_train)

# 5. Make predictions on the test set
y_pred = lr_model.predict(X_test_scaled)

# 6. Evaluate model performance
print("=== Logistic Regression Performance (Wine Dataset) ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 7. Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=target_names,
    yticklabels=target_names,
)
plt.title("Confusion Matrix - Logistic Regression (Wine)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()