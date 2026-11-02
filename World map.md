# Machine Learning Lab – External Viva Preparation Notes

11 Experiments

Code demonstration focused

Viva revision

Your main objective tomorrow is not just to remember the code. You should be able to explain:

1. What are you doing in this experiment?
2. Why did you use this particular library or function?
3. What happens when you execute each important line?
4. What output do you get?
5. What is the core ML concept behind it?

I'll follow your uploaded experiment codes, not replace them with unrelated implementations.

## First: common code structure

Almost all your experiments follow this general ML workflow:

1\. Import libraries

NumPy, Pandas, Matplotlib, Scikit-learn

2\. Load dataset

Load data into X (features) and y (target)

3\. Preprocessing

Handle missing values, encoding, scaling

4\. Train-test split

Separate training and testing data

5\. Model training

model.fit(X_train, y_train)

6\. Prediction and evaluation

predict(), accuracy, MSE, R², etc.

7\. Visualization

Graphs, charts and confusion matrices

Remember these three terms:

- `X` = Input features (independent variables).
- `y` = Target/output (dependent variable).
- `model.fit()` = Train the model using training data.

# Experiment 1 – Data Preprocessing and Feature Engineering



Titanic dataset

Supervised learning preprocessing



### 1. Aim

To perform data preprocessing and feature engineering using numerical and categorical data.

### 2. Code demonstration

Your code starts with:

```
import pandas as pdfrom sklearn.model_selection import train_test_splitfrom sklearn.impute import SimpleImputerfrom sklearn.preprocessing import OneHotEncoder, StandardScalerfrom sklearn.compose import ColumnTransformerfrom sklearn.pipeline import Pipeline
```

Why these libraries?

| Library/function  | Purpose                                         |
| ----------------- | ----------------------------------------------- |
| Pandas            | Data handling                                   |
| SimpleImputer     | Handle missing values                           |
| StandardScaler    | Scale numerical data                            |
| OneHotEncoder     | Convert categories into numbers                 |
| Pipeline          | Combine preprocessing steps                     |
| ColumnTransformer | Apply different operations to different columns |

### Step 1: Load dataset

```
df = pd.read_csv("Titanic-Dataset.csv")print(df.head())
```

- `read_csv()` reads CSV data.
- `head()` displays the first five rows.
- `df` is a DataFrame.

### Step 2: Separate features and target

```
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]y = df['Survived']
```

- X contains input features.
- y contains the output we want to predict.
- Survived is the target column.

### Step 3: Identify feature types

```
numerical_features = ['Age', 'Fare']categorical_features = ['Pclass', 'Sex', 'Embarked']
```

- Numerical: Values that represent quantities.
- Categorical: Values representing groups or categories.

### Step 4: Numerical preprocessing

```
numeric_transformer = Pipeline(steps=[    ('imputer', SimpleImputer(strategy='mean')),    ('scaler', StandardScaler())])
```

Two operations:

1. Missing Age/Fare values are replaced by their column mean.
2. StandardScaler transforms values into a common scale.

### Step 5: Categorical preprocessing

```
categorical_transformer = Pipeline(steps=[    ('imputer', SimpleImputer(strategy='most_frequent')),    ('encoder', OneHotEncoder(handle_unknown='ignore'))])
```

- Missing values are replaced with the most frequent category.
- OneHotEncoder converts categories into binary columns.

Example:

| Sex    | Male | Female |
| ------ | ---- | ------ |
| Male   | 1    | 0      |
| Female | 0    | 1      |

### Step 6: Combine transformations

```
preprocessor = ColumnTransformer(transformers=[    ('num', numeric_transformer, numerical_features),    ('cat', categorical_transformer, categorical_features)])
```

ColumnTransformer applies numerical and categorical preprocessing separately.

### Step 7: Split and transform

```
X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.2, random_state=42)X_train_processed = preprocessor.fit_transform(X_train)X_test_processed = preprocessor.transform(X_test)
```

- 80% training data.
- 20% testing data.
- `fit_transform()` learns preprocessing parameters and transforms training data.
- `transform()` applies the learned transformations to testing data.

### Core concepts to explain

What is data preprocessing?

Converting raw data into a clean and suitable format for machine learning.

What is feature engineering?

Creating, selecting or transforming input features to improve the quality of machine learning data.

What is data leakage?

When information from testing data incorrectly influences model training or preprocessing.

Most important viva question

Why do we use `fit_transform()` on training data and only `transform()` on testing data?

Answer: To prevent data leakage. The preprocessing parameters are learned only from training data and then applied to testing data.

# Experiment 2 – Dimensionality Reduction Using PCA and LDA



Digits dataset

Feature reduction



### 1. Aim

To compare PCA and LDA for dimensionality reduction and evaluate classification performance using Random Forest.

### 2. Code demonstration

### Step 1: Load dataset

```
from sklearn.datasets import load_digitsdata = load_digits()X, y = data.data, data.target
```

- Digits dataset contains handwritten digit images.
- Each image is 8 × 8 pixels.
- 8 × 8 = 64 features.
- Target represents digits 0–9.

### Step 2: Train-test split

```
X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.3, random_state=42)
```

70% training and 30% testing.

### Step 3: Standardization

```
scaler = StandardScaler()X_train_scaled = scaler.fit_transform(X_train)X_test_scaled = scaler.transform(X_test)
```

Standardization brings features to a comparable scale.

### Step 4: PCA

```
pca = PCA(n_components=15)X_train_pca = pca.fit_transform(X_train_scaled)X_test_pca = pca.transform(X_test_scaled)
```

PCA reduces 64 features to 15 features.

What is PCA?

Principal Component Analysis is an unsupervised dimensionality reduction technique that tries to preserve maximum variance in the data.

Understand variance with this example:

Imagine handwritten digit images. Some pixels change significantly between images, while other pixels remain almost the same.

PCA identifies directions in the data where the variation is greatest and represents the data using those directions.

### Step 5: LDA

```
lda = LDA(n_components=9)X_train_lda = lda.fit_transform(X_train_scaled, y_train)X_test_lda = lda.transform(X_test_scaled)
```

LDA reduces 64 features to 9 features.

What is LDA?

Linear Discriminant Analysis is a supervised dimensionality reduction technique that tries to maximize separation between different classes.

For 10 digit classes, maximum LDA components = 10 − 1 = 9.

### PCA vs LDA

| PCA                                     | LDA                              |
| --------------------------------------- | -------------------------------- |
| Unsupervised                            | Supervised                       |
| Does not use target labels              | Uses target labels               |
| Maximizes variance                      | Maximizes class separation       |
| Maximum components = number of features | Maximum components = classes − 1 |

### Step 6: Random Forest evaluation

```
model = RandomForestClassifier(random_state=42)model.fit(X_tr, y_tr)y_pred = model.predict(X_te)
```

The same Random Forest model is evaluated on:

- Original scaled data.
- PCA-reduced data.
- LDA-reduced data.

### Step 7: Evaluation

```
accuracy_score(y_te, y_pred)
```

Accuracy measures the percentage of correct predictions.

```
sum(pca.explained_variance_ratio_) * 100
```

Shows the percentage of total variance retained by the selected PCA components.

Most important viva question

What is the difference between PCA and LDA?

PCA preserves maximum information in terms of variance, while LDA tries to separate different classes.

# Experiment 3 – Multiple Linear Regression



Diabetes dataset

Regression



### 1. Aim

To implement Multiple Linear Regression and evaluate its performance using regression metrics.

### 2. Code demonstration

### Step 1: Load dataset

```
from sklearn.datasets import load_diabetesdata = load_diabetes()X = pd.DataFrame(data.data, columns=data.feature_names)y = data.target
```

- Diabetes is a built-in Scikit-learn dataset.
- It contains 442 samples and 10 input features.
- Target represents a quantitative measure of disease progression.

Why DataFrame?

```
X = pd.DataFrame(data.data, columns=data.feature_names)
```

`data.data` contains numerical values in an array.

`pd.DataFrame()` converts that array into a table and assigns meaningful column names to the features.

### Step 2: Train-test split

```
X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.2, random_state=42)
```

80% training and 20% testing.

### Step 3: Scaling

```
scaler = StandardScaler()X_train_scaled = scaler.fit_transform(X_train)X_test_scaled = scaler.transform(X_test)
```

StandardScaler standardizes input features.

### Step 4: Train model

```
model = LinearRegression()model.fit(X_train_scaled, y_train)
```

Linear Regression finds a relationship between input features and the target.

Mathematical equation:

\\[ Y=b_0+b_1X_1+b_2X_2+\cdots+b_nX_n \\]

Where:

- Y = Predicted output.
- b₀ = Intercept.
- b₁, b₂ = Coefficients.
- X₁, X₂ = Input features.

### Step 5: Prediction

```
y_pred = model.predict(X_test_scaled)
```

Predicts output for testing data.

### Step 6: Evaluation metrics

```
mse = mean_squared_error(y_test, y_pred)rmse = np.sqrt(mse)r2 = r2_score(y_test, y_pred)
```

| Metric | Meaning                                               |
| ------ | ----------------------------------------------------- |
| MSE    | Mean squared prediction error                         |
| RMSE   | Square root of MSE                                    |
| R²     | Proportion of target variation explained by the model |

Important:

- Lower MSE = Better.
- Lower RMSE = Better.
- R² closer to 1 = Better.

### Step 7: Visualization

```
plt.scatter(y_test, y_pred)plt.show()
```

Scatter plot compares actual and predicted values.

The red dashed line represents perfect predictions.

### Core concepts

What is regression?

Regression predicts continuous numerical values.

What is Multiple Linear Regression?

It predicts an output using two or more independent variables.

What is the difference between classification and regression?

Classification predicts categories; regression predicts numerical values.

# Experiment 4 – Polynomial Regression



California Housing dataset

Non-linear relationship



### 1. Aim

To implement Polynomial Regression using different polynomial degrees and compare their performance.

### 2. Code demonstration

### Step 1: Load dataset

```
from sklearn.datasets import fetch_california_housingdata = fetch_california_housing()X = data.data[:, [0]]y = data.target
```

- California Housing dataset contains housing information.
- `data.data[:, [0]]` selects the first feature, Median Income.
- `y` contains the target, Median House Value.

Note: Your code also contains a commented-out Diabetes dataset alternative using BMI. Your currently active code uses California Housing.

### Step 2: Train-test split

```
X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.2, random_state=42)
```

80% training and 20% testing.

### Step 3: Different polynomial degrees

```
degrees = [1, 2, 3, 5, 10]
```

We compare five polynomial degrees.

What is Polynomial Regression?

Polynomial Regression models a curved relationship between input and output using polynomial terms.

For example:

Linear:

\\[ y=b_0+b_1x \\]

Polynomial:

\\[ y=b_0+b_1x+b_2x^2+b_3x^3 \\]

### Step 4: Create model pipeline

```
model = make_pipeline(    PolynomialFeatures(degree=degree),    StandardScaler(),    LinearRegression())
```

Three operations:

1. `PolynomialFeatures()` generates polynomial features.
2. `StandardScaler()` scales the generated features.
3. `LinearRegression()` trains the regression model.

What does degree mean?

- Degree 1: Linear relationship.
- Degree 2: Quadratic relationship.
- Degree 3: Cubic relationship.
- Degree 10: Higher-order polynomial relationship.

### Step 5: Train and predict

```
model.fit(X_train, y_train)train_pred = model.predict(X_train)test_pred = model.predict(X_test)
```

Train the model and predict outputs for both datasets.

### Step 6: Calculate errors

```
train_mse = mean_squared_error(y_train, train_pred)test_mse = mean_squared_error(y_test, test_pred)
```

Compare training and testing errors for each degree.

### Step 7: Plot graph

```
plt.plot(degrees, train_errors, marker='o')plt.plot(degrees, test_errors, marker='s')plt.show()
```

Displays training and testing MSE against polynomial degree.

### Core concepts

What is overfitting?

When a model learns training data too closely, including noise, and performs poorly on unseen data.

What is underfitting?

When a model is too simple to capture the actual relationship in the data.

What is the bias-variance trade-off?

It is the balance between a model being too simple and too sensitive to training data.

Most important viva question

What happens when polynomial degree increases?

The model becomes more flexible and can capture complex relationships, but a very high degree can cause overfitting.

# Experiment 5 – Ridge and Lasso Regression



Diabetes dataset

Regularization



### 1. Aim

To implement and compare Linear, Ridge and Lasso Regression.

### 2. Code demonstration

### Step 1: Load dataset

```
data = load_diabetes(as_frame=True)X = data.datay = data.target
```

`as_frame=True` loads the data in Pandas DataFrame format.

### Step 2: Train-test split

```
X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.3, random_state=42)
```

70% training and 30% testing.

### Step 3: Feature scaling

```
scaler = StandardScaler()X_train = scaler.fit_transform(X_train)X_test = scaler.transform(X_test)
```

Scaling is important because regularization penalizes coefficient sizes.

### Step 4: Create models

```
models = {    "Linear Regression": LinearRegression(),    "Ridge Regression": Ridge(alpha=1.0),    "Lasso Regression": Lasso(alpha=0.1, max_iter=10000)}
```

Three models are created.

Linear Regression: Fits a linear relationship without a regularization penalty.

Ridge Regression: Uses L2 regularization. It reduces large coefficients but generally does not make them exactly zero.

Lasso Regression: Uses L1 regularization. It can make some coefficients exactly zero, helping with feature selection.

### Step 5: Train models

```
for name, model in models.items():    model.fit(X_train, y_train)    train_pred = model.predict(X_train)    test_pred = model.predict(X_test)
```

A loop trains all three models one by one.

### Step 6: Evaluation

```
train_r2 = r2_score(y_train, train_pred)test_r2 = r2_score(y_test, test_pred)train_rmse = mean_squared_error(    y_train, train_pred) ** 0.5
```

The code calculates R² and RMSE for training and testing data.

### Step 7: Comparison

```
plt.bar(result_df["Model"], result_df["Test R2"])plt.show()
```

Bar chart compares test R² scores of all three models.

### Ridge vs Lasso

| Ridge                        | Lasso                      |
| ---------------------------- | -------------------------- |
| L2 regularization            | L1 regularization          |
| Squares coefficients         | Uses absolute coefficients |
| Shrinks coefficients         | Can make coefficients zero |
| Usually retains all features | Performs feature selection |

What is regularization?

Regularization adds a penalty to the model to control large coefficients and reduce overfitting.

What is alpha?

Alpha controls the strength of regularization. Higher alpha means stronger penalty.

Most important viva question

Why do we use Ridge and Lasso?

To reduce overfitting by penalizing large model coefficients.

# Experiment 6 – Logistic Regression



Wine dataset

Binary classification



### 1. Aim

To implement Logistic Regression for binary classification and evaluate its performance.

### 2. Code demonstration

### Step 1: Load dataset

```
from sklearn.datasets import load_winewine = load_wine()X = wine.data
```

Wine dataset contains 178 samples and 13 features.

### Step 2: Convert into binary classification

```
y = (wine.target == 0).astype(int)target_names = ["Not Class 0", "Class 0"]
```

This is an important line.

Original Wine dataset has three classes: 0, 1 and 2.

Here we convert it into two classes:

- Class 0 → 1
- Class 1 and 2 → 0

`astype(int)` converts Boolean values into integers.

### Step 3: Split dataset

```
X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.2,    random_state=100, stratify=y)
```

- 80% training.
- 20% testing.
- `stratify=y` maintains class proportions in both sets.

### Step 4: Standardization

```
scaler = StandardScaler()X_train_scaled = scaler.fit_transform(X_train)X_test_scaled = scaler.transform(X_test)
```

Scaling helps Logistic Regression converge effectively.

### Step 5: Model training

```
lr_model = LogisticRegression(random_state=100)lr_model.fit(X_train_scaled, y_train)
```

What is Logistic Regression?

Logistic Regression is a supervised classification algorithm that predicts the probability of a class.

It uses the sigmoid function:

\\[ \sigma(z)=\frac{1}{1+e^{-z}} \\]

Output is between 0 and 1.

### Step 6: Prediction

```
y_pred = lr_model.predict(X_test_scaled)
```

Predicts class labels.

### Step 7: Evaluation

```
accuracy_score(y_test, y_pred)classification_report(    y_test, y_pred,    target_names=target_names)
```

Classification report shows:

- Precision
- Recall
- F1-score
- Support

### Step 8: Confusion matrix

```
cm = confusion_matrix(y_test, y_pred)sns.heatmap(cm, annot=True, fmt="d")plt.show()
```

Confusion matrix compares actual and predicted classes.

| Term | Meaning                        |
| ---- | ------------------------------ |
| TP   | Correctly predicted positive   |
| TN   | Correctly predicted negative   |
| FP   | Incorrectly predicted positive |
| FN   | Incorrectly predicted negative |

### Core concepts

Why is Logistic Regression used for classification?

Because it predicts class probabilities and assigns classes based on a decision threshold.

What is sigmoid function?

It converts any real-valued input into a value between 0 and 1.

What is the default classification threshold?

0.5.

# Experiment 7 – Support Vector Machine (SVM)



Breast Cancer dataset

Classification



### 1. Aim

To implement SVM using different kernels and compare classification performance.

### 2. Code demonstration

### Step 1: Load dataset

```
from sklearn.datasets import load_breast_cancerdata = load_breast_cancer()X, y = data.data, data.target
```

Breast Cancer dataset contains 569 samples and 30 features.

It has two classes:

- Malignant
- Benign

### Step 2: Split data

```
X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.2,    random_state=42, stratify=y)
```

80% training and 20% testing.

### Step 3: Standardization

```
scaler = StandardScaler()X_train_s = scaler.fit_transform(X_train)X_test_s = scaler.transform(X_test)
```

Scaling is particularly important for SVM because it relies on distances and margins.

### Step 4: Define kernels

```
kernels = {    "Linear": SVC(kernel='linear', C=1.0),    "Polynomial": SVC(        kernel='poly', degree=3, C=1.0    ),    "RBF": SVC(        kernel='rbf', C=1.0, gamma='scale'    )}
```

Three kernels are compared.

What is SVM?

Support Vector Machine is a supervised learning algorithm that finds an optimal decision boundary to separate different classes.

What is a kernel?

A kernel helps SVM model different types of relationships between data points.

### Three kernels

[Linear Support Vector Machines Explained | by Mazen Ahmed | Medium](https://images.openai.com/static-rsc-4/7mGiOyYl6kFa4nIs1iu4AJvN6CVCBeEesrFZ9TsbwG0ipn-XT1J_u26Y6LF3BnHf5XBBg-8dGD3CnouDsEADdBjzjbwm5HU1rW9Zr-oJZAm1I8UX32dkL1UQ0IWa4V2CILJLc37O0SeI5Bruz9CdtV5LqN_Jr7K3pCw9ujjo-YU?purpose=inline)

1\. Linear Kernel

Separates classes using a straight decision boundary.

[Support Vector Machines Explained Simply | Medium](https://images.openai.com/static-rsc-4/1vzYpVOUuNAhb50qRB1JzXwt1Nn-GQMkZZeJW2tTm-c_IIpXCo_jCpdzayorGYCbCaPrJkq4t4FdeMn5PcBMmyibdo_SMpICNiWH98tqwHkzGcJzqhDNx0_DLo5ZI2UIYX32yS5yniBPX1gkNDbMiPnW5Yf20VdgfpLJvokQ2p8?purpose=inline)

2\. Polynomial Kernel

Uses polynomial relationships to create a curved decision boundary.

[Why is “Kernel Trick” called a “Trick” ? | by shubhankar Keshri | Medium](https://images.openai.com/static-rsc-4/RLTDaRdrNhlZyOMn_7Tq2X1PL68zjlQIBwlruXwAXJOopdvNDbikDfnA8I1Jn4Mtwmvi71lHDCo_CEhaiTPY0b47JA2azzHvpt3UwnS2VYn5t_Ayt2Liqk200sUKvpmUQfPvoIntNkXeTYyictb88AdNYD0qqtNssWWr0HLALNw?purpose=inline)

3\. RBF Kernel

Handles complex nonlinear relationships using radial basis functions.

### Step 5: Train and predict

```
for name, model in kernels.items():    model.fit(X_train_s, y_train)    y_pred = model.predict(X_test_s)
```

Loop trains and evaluates each kernel.

### Step 6: Evaluation metrics

```
acc = accuracy_score(y_test, y_pred)prec = precision_score(y_test, y_pred)rec = recall_score(y_test, y_pred)f1 = f1_score(y_test, y_pred)
```

| Metric    | Meaning                                                |
| --------- | ------------------------------------------------------ |
| Accuracy  | Overall correct predictions                            |
| Precision | Correct positive predictions among predicted positives |
| Recall    | Correct positive predictions among actual positives    |
| F1-score  | Harmonic mean of precision and recall                  |

### Step 7: Best kernel

```
best = result_df.loc[    result_df["Accuracy"].idxmax()]
```

Finds the kernel with the highest accuracy.

### Core concepts

What are support vectors?

Data points closest to the decision boundary. They help determine the position of the boundary.

What is C in SVM?

C controls the penalty for classification errors. Higher C penalizes errors more strongly.

What is gamma in RBF?

Gamma controls the influence of individual training samples. Higher gamma produces more localized influence.

What is the objective of SVM?

To find a decision boundary with maximum margin between classes.

# Experiment 8 – K-Means Clustering



Iris dataset

Unsupervised learning



### 1. Aim

To implement K-Means clustering and determine the optimal number of clusters using Elbow and Silhouette methods.

### 2. Code demonstration

### Step 1: Load dataset

```
from sklearn.datasets import load_irisiris = load_iris()X = iris.data
```

Iris dataset contains:

- 150 samples.
- 4 features.
- 3 flower species.

Here we use only the input features. No target labels are given to K-Means.

### Step 2: Scaling

```
X_scaled = StandardScaler().fit_transform(X)
```

Standardizes all four features.

### Step 3: Define cluster range

```
k_range = range(2, 11)k_all = range(1, 11)
```

We evaluate cluster numbers from 1 to 10.

### Step 4: Apply K-Means

```
for k in k_all:    kmeans = KMeans(        n_clusters=k,        init='k-means++',        random_state=42,        n_init=10    )    labels = kmeans.fit_predict(X_scaled)    wcss_all.append(kmeans.inertia_)
```

Important parameters:

| Parameter    | Meaning                        |
| ------------ | ------------------------------ |
| n_clusters   | Number of clusters             |
| init         | Method to initialize centroids |
| random_state | Reproducible results           |
| n_init       | Number of initializations      |

`fit_predict()` trains the clustering model and returns cluster labels.

### Step 5: WCSS / Elbow method

```
kmeans.inertia_
```

Inertia represents WCSS (Within-Cluster Sum of Squares).

It measures the sum of squared distances between data points and their assigned cluster centers.

What is Elbow Method?

We plot WCSS against the number of clusters.

As K increases, WCSS decreases. We look for the point where the decrease starts slowing down significantly. This is called the elbow point.

[K-means Clustering: Complete Guide with Algorithm, Implementation & Best Practices - Interactive | Michael Brenndoerfer | Michael Brenndoerfer](https://images.openai.com/static-rsc-4/7KyAP-bU3wtlC7c-QVeLJZb1Bo94ga0YknB4Yf_seilBHlIB3PJcg2rAKznqI9bS_6JjJmP8kwX0bGaWz5CeRfbdYG0D2fkHMsVDAb5TiPRaA4EWR6Bv96ODpJ2i4fiubpsADShECmU859pmxKh61u0eohApuKJnYqr1jZbJaUs?purpose=inline)

[mbrenndoerfer.com](https://mbrenndoerfer.com/writing/kmeans-clustering-complete-guide?utm_source=chatgpt.com)

### Step 6: Silhouette score

```
sil_score = silhouette_score(X_scaled, labels)
```

Silhouette Score measures how well a data point fits within its own cluster compared with other clusters.

Range:

\\[ -1 \leq S \leq 1 \\]

- Near +1: Good clustering.
- Around 0: Overlapping clusters.
- Negative: Possibly incorrect cluster assignment.

### Step 7: Find optimal K

```
optimal_k = list(k_range)[    silhouette_scores.index(max(silhouette_scores))]
```

Selects the K having the highest Silhouette Score.

### Step 8: Final clustering

```
final_kmeans = KMeans(    n_clusters=optimal_k,    init='k-means++',    random_state=42,    n_init=10)cluster_labels = final_kmeans.fit_predict(X_scaled)
```

Performs final clustering using the selected K.

### Core concepts

What is clustering?

Grouping similar data points together without predefined labels.

What is centroid?

The center point of a cluster, calculated using the mean of its data points.

What is K-Means?

An unsupervised algorithm that divides data into K clusters based on distance from cluster centroids.

What is the difference between Elbow and Silhouette methods?

- Elbow uses WCSS.
- Silhouette evaluates cluster separation and compactness.

Why do we use K-Means++?

It helps select better initial centroids and can improve clustering convergence.

# Experiment 9 – DBSCAN Clustering



Iris dataset

Density-based clustering



### 1. Aim

To implement DBSCAN clustering and detect clusters and noise points without specifying the number of clusters beforehand.

### 2. Code demonstration

### Step 1: Load dataset

```
iris = load_iris()X = iris.data
```

Same Iris dataset with 150 samples and 4 features.

### Step 2: Scaling

```
X_scaled = StandardScaler().fit_transform(X)
```

Scaling is important because DBSCAN uses distance to identify neighboring points.

### Step 3: Apply DBSCAN

```
eps_val = 0.6min_samples_val = 5dbscan = DBSCAN(    eps=eps_val,    min_samples=min_samples_val)labels = dbscan.fit_predict(X_scaled)
```

Two important parameters:

1\. eps (Epsilon)

Maximum distance between two points for considering them neighbors.

2\. min_samples

Minimum number of nearby samples required to form a dense region.

### Step 4: Identify clusters and noise

```
unique_labels = set(labels)n_clusters = len(unique_labels - {-1})n_noise = np.sum(labels == -1)
```

Important:

- `-1` represents noise points.
- `0, 1, 2...` represent cluster labels.

### Step 5: Cluster distribution

```
for cluster_id in sorted(unique_labels):    count = np.sum(labels == cluster_id)
```

Counts how many samples belong to each cluster or noise category.

### Step 6: Silhouette score

```
non_noise_mask = (labels != -1)if len(set(labels[non_noise_mask])) > 1:    sil_score = silhouette_score(        X_scaled[non_noise_mask],        labels[non_noise_mask]    )
```

Calculates Silhouette Score after excluding noise points.

The condition ensures there are at least two non-noise clusters before calculating the score.

### Step 7: PCA visualization

```
pca = PCA(n_components=2)X_pca = pca.fit_transform(X_scaled)
```

PCA reduces four features into two components for visualization.

It does not change the original DBSCAN clustering. Clustering was already performed using all four scaled features.

### Core concepts

What is DBSCAN?

Density-Based Spatial Clustering of Applications with Noise.

It groups closely packed data points and identifies isolated points as noise.

What are the three types of DBSCAN points?

1. Core point: Has at least min_samples points in its epsilon neighborhood, including itself.
2. Border point: Lies within the neighborhood of a core point but does not itself satisfy the core-point density requirement.
3. Noise point: Neither a core point nor reachable from a core point.

[Clustering Isn’t Just Grouping — It’s Decision Making](https://images.openai.com/static-rsc-4/XP23snn_CAZTuJuN1Bk739Gm_1fVc1KxYOV2codNVvir2gARQF-77250lsIEgcumZUwxjdOsigWvleDqhzdjRftZEBfH13yLYA8EH-wEExCupWbbQFVeUag-ROPWhEKB9El2LlntSVEjoEnt2qYt51EpXfgaA2oCGc7uNCSHsCg?purpose=inline)

[codeandclarity.substack.com](https://codeandclarity.substack.com/p/clustering-isnt-just-grouping-its?utm_source=chatgpt.com)

### K-Means vs DBSCAN

| K-Means                             | DBSCAN                                 |
| ----------------------------------- | -------------------------------------- |
| Requires K beforehand               | Does not require K                     |
| Centroid-based                      | Density-based                          |
| Sensitive to outliers               | Identifies noise                       |
| Best for roughly spherical clusters | Can identify irregular-shaped clusters |

Most important viva question

What is the difference between K-Means and DBSCAN?

K-Means requires the number of clusters in advance and uses centroids, while DBSCAN groups points based on density and can identify noise.

# Experiment 10 – Ensemble Learning



Breast Cancer dataset

Decision Tree, Random Forest, Gradient Boosting



### 1. Aim

To compare Decision Tree, Random Forest and Gradient Boosting classifiers using ensemble learning techniques.

### 2. Code demonstration

### Step 1: Load dataset

```
data = load_breast_cancer()X = pd.DataFrame(    data.data,    columns=data.feature_names)y = data.target
```

- 569 samples.
- 30 features.
- Two classes: Malignant and Benign.

### Step 2: Train-test split

```
X_train, X_test, y_train, y_test = train_test_split(    X, y,    test_size=0.2,    random_state=RANDOM_STATE,    stratify=y)
```

80% training and 20% testing.

### Step 3: Define models

```
models = {    "Baseline (Decision Tree)":        DecisionTreeClassifier(random_state=42),    "Random Forest (Bagging)":        RandomForestClassifier(            n_estimators=100,            random_state=42        ),    "Gradient Boosting (Boosting)":        GradientBoostingClassifier(            n_estimators=100,            random_state=42        )}
```

Three algorithms are compared.

1\. Decision Tree

A tree-like model that makes decisions using feature-based conditions.

2\. Random Forest

Uses multiple decision trees and combines their predictions.

3\. Gradient Boosting

Builds models sequentially, with each new model attempting to correct errors made by previous models.

### Step 4: Train and predict

```
for name, model in models.items():    model.fit(X_train, y_train)    y_pred = model.predict(X_test)    y_proba = model.predict_proba(X_test)[:, 1]
```

- `fit()` trains the model.
- `predict()` predicts class labels.
- `predict_proba()` gives class probabilities.
- `[:, 1]` selects probabilities of class 1.

### Step 5: Calculate metrics

```
acc = accuracy_score(y_test, y_pred)prec = precision_score(y_test, y_pred)rec = recall_score(y_test, y_pred)f1 = f1_score(y_test, y_pred)auc = roc_auc_score(y_test, y_proba)
```

| Metric    | Meaning                                |
| --------- | -------------------------------------- |
| Accuracy  | Overall correct predictions            |
| Precision | Correct positive predictions           |
| Recall    | Ability to identify actual positives   |
| F1-score  | Balance between precision and recall   |
| ROC-AUC   | Ability to distinguish between classes |

### Step 6: Feature importance

```
feature_importances = pd.DataFrame({    'Feature': X.columns,    'Random Forest': rf_model.feature_importances_,    'Gradient Boosting': gb_model.feature_importances_})
```

Feature importance identifies which input features contribute most to the tree-based model's predictions.

### Step 7: Visualization

```
sns.barplot(    data=df_melted,    x="Metric",    y="Score",    hue="Model")
```

Compares performance metrics across all three models.

Another graph compares the top 10 important features.

### Core concepts

What is Ensemble Learning?

Combining multiple machine learning models to improve prediction performance.

What is Bagging?

Training multiple models independently on different bootstrap samples and combining their predictions.

Random Forest uses Bagging.

What is Boosting?

Training models sequentially, where later models focus on correcting earlier errors.

Gradient Boosting uses Boosting.

### Random Forest vs Gradient Boosting

| Random Forest                   | Gradient Boosting              |
| ------------------------------- | ------------------------------ |
| Uses Bagging                    | Uses Boosting                  |
| Trees are trained independently | Trees are trained sequentially |
| Reduces variance                | Builds on previous errors      |
| Combines tree predictions       | Adds tree contributions        |

What is n_estimators?

Number of trees or boosting stages in the model. In your code, it is 100.

What is ROC-AUC?

It measures how well a classifier distinguishes between positive and negative classes. Higher values indicate better discrimination.

Most important viva question

What is the difference between Bagging and Boosting?

Bagging trains models independently and combines results. Boosting trains models sequentially, focusing on previous errors.

# Experiment 11 – Reinforcement Learning Using Q-Learning



4 × 4 Grid World

Reinforcement Learning



### 1. Aim

To implement Q-Learning in a Grid World environment and learn an optimal policy using rewards and penalties.

### 2. Understand the environment first

Your code creates a 4 × 4 grid.

## Grid World

START

⬜

⬜

⬜

⬜

⬜

⬜

⬜

⬜

⬜

⚠️

⬜

⬜

⬜

⬜

GOAL

🔵 Start: (0,0)

🔴 Hazard: (2,2)

🟢 Goal: (3,3)

The agent starts at (0,0) and tries to reach (3,3) while avoiding the hazard at (2,2).

### Step 1: Import libraries

```
import randomimport numpy as npimport matplotlib.pyplot as plt
```

- Random: Random action selection.
- NumPy: Q-table creation and calculations.
- Matplotlib: Learning curve visualization.

### Step 2: Define environment

```
ROWS, COLS = 4, 4START = (0, 0)GOAL = (3, 3)HAZARD = (2, 2)ACTIONS = [    (-1, 0), (1, 0),    (0, -1), (0, 1)]
```

Four actions:

- Up
- Down
- Left
- Right

### Step 3: Convert position into state

```
def to_state(r, c):    return r * COLS + c
```

Converts 2D grid coordinates into a single state number.

Example:

| Position | State |
| -------- | ----- |
| (0,0)    | 0     |
| (0,1)    | 1     |
| (1,0)    | 4     |
| (3,3)    | 15    |

Total states = 4 × 4 = 16.

### Step 4: Define step function

```
def step(pos, action_idx):    dr, dc = ACTIONS[action_idx]    nr = min(max(pos[0] + dr, 0), ROWS - 1)    nc = min(max(pos[1] + dc, 0), COLS - 1)    nxt = (nr, nc)
```

This calculates the next position while keeping the agent inside the grid.

Now the reward conditions:

```
if nxt == GOAL:    return nxt, 10.0, Trueif nxt == HAZARD:    return nxt, -10.0, Truereturn nxt, -1.0, False
```

| Situation       | Reward |
| --------------- | ------ |
| Reaches goal    | +10    |
| Enters hazard   | -10    |
| Normal movement | -1     |

`True` means the episode is finished.

### Step 5: Define Q-Learning parameters

```
alpha = 0.1gamma = 0.95episodes = 1000eps_start = 1.0eps_end = 0.05
```

| Parameter | Meaning                   |
| --------- | ------------------------- |
| Alpha     | Learning rate             |
| Gamma     | Discount factor           |
| Episodes  | Number of training rounds |
| Epsilon   | Exploration probability   |

Alpha: Controls how much new information changes existing knowledge.

Gamma: Controls the importance of future rewards.

Epsilon: Controls exploration versus exploitation.

### Step 6: Create Q-table

```
Q = np.zeros((ROWS * COLS, len(ACTIONS)))rewards_history = []
```

Q-table shape:

\\[ 16 \times 4 \\]

- 16 rows represent states.
- 4 columns represent actions.
- Initially all Q-values are zero.

What is Q-value?

It represents the expected future reward for taking a particular action in a particular state.

### Step 7: Training loop

```
for ep in range(episodes):    pos = START    eps = eps_end + (        eps_start - eps_end    ) * np.exp(-0.005 * ep)
```

- Runs 1000 training episodes.
- Every episode starts from the starting position.
- Epsilon gradually decreases.

### Step 8: Exploration vs exploitation

```
if random.random() < eps:    action = random.randrange(len(ACTIONS))else:    action = int(np.argmax(Q[s]))
```

Two strategies:

Exploration: Agent tries random actions to discover new possibilities.

Exploitation: Agent chooses the action with the highest known Q-value.

### Step 9: Q-Learning update rule

This is the most important part of Experiment 11.

```
target = reward if done else (    reward + gamma * np.max(Q[ns]))Q[s, action] += alpha * (    target - Q[s, action])
```

Mathematical formula:

\\[ Q(s,a)\leftarrow Q(s,a)+ \alpha[r+\gamma\max Q(s',a')-Q(s,a)] \\]

For terminal states, the future reward term is zero.

Understand each term:

| Symbol       | Meaning             |
| ------------ | ------------------- |
| Q(s,a)       | Current Q-value     |
| α            | Learning rate       |
| r            | Immediate reward    |
| γ            | Discount factor     |
| s'           | Next state          |
| max Q(s',a') | Best future Q-value |

In simple words:

The agent updates its knowledge based on the reward it receives and the best possible future reward.

### Step 10: Display learned policy

```
best_act = int(np.argmax(Q[to_state(r, c)]))
```

Selects the action with the highest Q-value for each state.

The code displays arrows showing the learned policy.

### Step 11: Evaluate greedy policy

```
action = int(np.argmax(Q[to_state(*pos)]))pos, reward, done = step(pos, action)
```

The trained agent follows the best-known action instead of choosing random actions.

### Step 12: Learning curve

```
moving_avg = np.convolve(    rewards_history,    np.ones(window_size)/window_size,    mode='valid')
```

Calculates a 50-episode moving average of rewards.

The graph shows how the agent's reward changes during training.

### Core concepts

What is Reinforcement Learning?

A type of machine learning where an agent learns by interacting with an environment and receiving rewards or penalties.

What is an agent?

The learner that takes actions.

What is an environment?

The world in which the agent operates.

What is a policy?

A strategy that tells the agent which action to take in each state.

What is Q-Learning?

A model-free, off-policy reinforcement learning algorithm that learns the value of state-action pairs.

What is the difference between supervised and reinforcement learning?

Supervised learning learns from labelled data. Reinforcement learning learns through actions, rewards and penalties.

Most important viva question

Explain Q-Learning in simple words.

Q-Learning is an algorithm where an agent learns the best action to take in each state by receiving rewards and penalties from the environment.

# Last-minute viva revision – All 11 experiments

These are the questions an external examiner can ask across multiple experiments.

## 1. Most important Python functions

| Function               | What it does                   |
| ---------------------- | ------------------------------ |
| `read_csv()`           | Reads CSV file                 |
| `head()`               | Displays first five rows       |
| `fit()`                | Trains a model                 |
| `transform()`          | Transforms data                |
| `fit_transform()`      | Fits and transforms data       |
| `predict()`            | Predicts output                |
| `train_test_split()`   | Divides dataset                |
| `StandardScaler()`     | Standardizes features          |
| `accuracy_score()`     | Calculates accuracy            |
| `mean_squared_error()` | Calculates MSE                 |
| `r2_score()`           | Calculates R²                  |
| `plt.show()`           | Displays graph                 |
| `np.argmax()`          | Returns index of maximum value |
| `pd.DataFrame()`       | Creates a tabular structure    |

## 2. Common viva questions

Q1. What is Machine Learning?

Machine Learning is a branch of AI that enables computers to learn from data and make predictions without explicitly programming every rule.

Q2. What are the three types of Machine Learning?

1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

Q3. What is supervised learning?

Learning using labelled data.

Q4. What is unsupervised learning?

Learning patterns from unlabelled data.

Q5. What is classification?

Predicting categories, such as benign or malignant.

Q6. What is regression?

Predicting continuous numerical values, such as house prices.

Q7. What is clustering?

Grouping similar data points together.

Q8. Why do we split datasets?

To train the model on one part and evaluate its performance on unseen data.

Q9. What is random_state?

It ensures reproducibility by controlling random splitting or initialization.

Q10. What is overfitting?

When a model performs well on training data but poorly on unseen data.

Q11. What is underfitting?

When a model is too simple to learn the underlying patterns.

Q12. What is normalization vs standardization?

- Normalization generally scales values to a fixed range, often 0 to 1.
- Standardization transforms values to mean 0 and standard deviation 1.

Q13. What is NumPy?

A Python library for numerical calculations and arrays.

Q14. What is Pandas?

A Python library for data manipulation and analysis.

Q15. What is Scikit-learn?

A Python library providing machine learning algorithms and preprocessing tools.

## 3. Important formulas to memorize

| Concept   | Formula                                       |
| --------- | --------------------------------------------- |
| Accuracy  | (TP + TN) / Total                             |
| Precision | TP / (TP + FP)                                |
| Recall    | TP / (TP + FN)                                |
| F1-score  | 2 × Precision × Recall / (Precision + Recall) |
| MSE       | Mean of squared errors                        |
| RMSE      | √MSE                                          |
| R²        | 1 − (SSres / SStot)                           |
| Sigmoid   | 1 / (1 + e⁻ᶻ)                                 |

## 4. Important code corrections and warnings

I want you to know these before tomorrow rather than discovering them during the demonstration.

1. Experiment 1: Titanic CSV is required by your original code. You said to skip downloading it, so arrange the CSV separately if you intend to demonstrate Experiment 1 exactly.
2. Experiment 4: Your active code uses `fetch_california_housing()`. It requires downloading the dataset. Your commented Diabetes alternative works offline.
3. Experiment 2: LDA uses `y_train` during fitting because it is supervised. PCA does not.
4. Experiment 9: DBSCAN labels noise as `-1`, not as an ordinary cluster.
5. Experiment 11: Q-Learning's future reward is excluded when the episode reaches a terminal state.

# Your code demonstration strategy tomorrow

For every experiment, practise explaining these five things in order:

1. Dataset: Which dataset did you use and how many features does it contain?
2. Preprocessing: Did you apply scaling, encoding or dimensionality reduction?
3. Algorithm: Which model did you implement and why?
4. Training: Explain `fit()`, `predict()` and important parameters.
5. Output: Explain the evaluation metrics and graph.