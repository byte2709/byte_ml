import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

"""
data = {
    'Pclass': [1, 3, 2, 1, 3, 2, 1, 3, 2, 1],
    'Sex': ['male', 'female', 'male', 'female', 'male',
            'female', 'male', 'female', 'male', 'female'],
    'Age': [22, 25, np.nan, 35, 28, 30, 45, np.nan, 19, 40],
    'Fare': [7.25, 71.8, 8.05, np.nan, 7.9, 13.0, 30.0, 15.5, np.nan, 50.0],
    'Embarked': ['S', 'C', 'S', 'Q', 'S', np.nan, 'C', 'S', 'Q', 'C'],
    'Survived': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

print("Original Dataset:")
print(df.head())
"""

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")
print("Original Dataset:")
print(df.head())

# Select features and target
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = df['Survived']

# Identify numerical and categorical columns
numerical_features = ['Age', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

# Handle missing values and scale numerical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Handle missing values and encode categorical feaures
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply feature engineering
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Display results
print("\nTraining data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)
print("\nFeature engineering completed successfully.")