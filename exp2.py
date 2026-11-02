import time
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_digits()
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA (64 features reduced to 15)
pca = PCA(n_components=15)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Apply LDA (64 features reduced to 9)
lda = LDA(n_components=9)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

# Evaluate model performance
def evaluate(X_tr, X_te, y_tr, y_te, method):
    model = RandomForestClassifier(random_state=42)

    start = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - start

    start = time.time()
    y_pred = model.predict(X_te)
    predict_time = time.time() - start

    return {
        "Method": method,
        "Features": X_tr.shape[1],
        "Accuracy (%)": round(accuracy_score(y_te, y_pred) * 100, 2),
        "Training Time (s)": round(train_time, 4),
        "Prediction Time (s)": round(predict_time, 4)
    }

# Compare original data, PCA and LDA
results = [
    evaluate(X_train_scaled, X_test_scaled, y_train, y_test,
             "Original Data"),
    evaluate(X_train_pca, X_test_pca, y_train, y_test,
             "PCA Reduced"),
    evaluate(X_train_lda, X_test_lda, y_train, y_test,
             "LDA Reduced")
]

print("\nDimensionality Reduction Results:")
print(pd.DataFrame(results).to_string(index=False))
print("\nPCA variance retained:",
    round(sum(pca.explained_variance_ratio_) * 100, 2), "%")