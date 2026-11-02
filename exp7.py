import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Feature scaling (SVM is distance-based, scaling is mandatory)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Step 4: Define SVM kernels (linear and nonlinear)
kernels = {
    "Linear": SVC(kernel='linear', C=1.0),
    "Polynomial": SVC(kernel='poly', degree=3, C=1.0),
    "RBF": SVC(kernel='rbf', C=1.0, gamma='scale'),
}

results = []

# Step 5: Train, predict, and evaluate each kernel
for name, model in kernels.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    results.append({"Kernel": name, "Accuracy": round(acc, 4),
                    "Precision": round(prec, 4), "Recall": round(rec, 4), "F1-Score": round(f1, 4)})
    
    print(f"\n{'=' * 25}")
    print(f"Kernel: {name}")
    print(f"{'=' * 25}")
    print(f"Accuracy : {acc:.4f}  \nPrecision: {prec:.4f}  \nRecall: {rec:.4f}  \nF1: {f1:.4f}")
    print("\nConfusion Matrix:\n", cm)

# Step 6: Comparison table
result_df = pd.DataFrame(results)
print("\nKernel Comparison:\n", result_df.to_string(index=False))

# Step 7: Bar chart comparison
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(result_df))
width = 0.2

fig, ax = plt.subplots(figsize=(9, 5))
for i, metric in enumerate(metrics):
    ax.bar(x + i * width, result_df[metric], width, label=metric)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(result_df["Kernel"])
ax.set_ylabel("Score")
ax.set_title("SVM Kernel Performance Comparison")
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Step 8: Best kernel
best = result_df.loc[result_df["Accuracy"].idxmax()]
print(f"\nBest Kernel: {best['Kernel']} (Accuracy = {best['Accuracy']})")