import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Set random seed for reproducibility
RANDOM_STATE = 42

# 1. Load Dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(f"Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features.")
print(f"Target Distribution: {np.bincount(y)} (0: Malignant, 1: Benign)\n")

# 2. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 3. Define Models (Baseline vs. Ensembles)
models = {
    "Baseline (Decision Tree)": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest (Bagging)": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "Gradient Boosting (Boosting)": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
}

# 4. Train and Evaluate Models
results = []

for name, model in models.items():
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    })

# 5. Display Comparison Table
df_results = pd.DataFrame(results)
print("=== Model Performance Comparison ===")
print(df_results.to_string(index=False))

# 6. Feature Importance Comparison (Random Forest vs Gradient Boosting)
rf_model = models["Random Forest (Bagging)"]
gb_model = models["Gradient Boosting (Boosting)"]

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Random Forest': rf_model.feature_importances_,
    'Gradient Boosting': gb_model.feature_importances_
}).sort_values(by='Random Forest', ascending=False)

# 7. Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Performance Metrics Comparison
df_melted = df_results.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", ax=axes[0], palette="viridis")
axes[0].set_title("Performance Metrics across Models")
axes[0].set_ylim(0.85, 1.0)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Plot 2: Top 10 Feature Importances
top_features = feature_importances.head(10).melt(id_vars="Feature", var_name="Model", value_name="Importance")
sns.barplot(data=top_features, x="Importance", y="Feature", hue="Model", ax=axes[1], palette="magma")
axes[1].set_title("Top 10 Feature Importances")

plt.tight_layout()
plt.show()
