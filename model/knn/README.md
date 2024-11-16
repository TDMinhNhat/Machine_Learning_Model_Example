# KNN (K-NEAREST-NEIGHBORS) 

K-Nearest Neighbors is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions). KNN has been used in statistical estimation and pattern recognition already in the beginning of 1970’s as a non-parametric technique. Algorithm is based on the feature similarity approach.

## How does the algorithm work?

1. Pick a value for K.
2. Calculate the distance of unknown case from all cases.
3. Select the K-observations in the training data that are "nearest" to the unknown data point.
4. Predict the response of the unknown data point using the most popular response value from the K-nearest neighbors.
5. Done!
6. The distance can be of any type e.g Euclidean or Manhattan etc.
7. The value of K can be found by using cross-validation.
8. The algorithm is not suitable for large data sets as it requires more time to compute the distance of each point from all the other points.
9. The algorithm is not suitable for high dimensional data as it is difficult to find the distance in each dimension.
10. The algorithm is not suitable for categorical data as it is difficult to find the distance in each category.
11. The algorithm is not suitable for missing data as it requires all the values to compute the distance.
12. The algorithm is not suitable for imbalanced data as it requires more time to compute the distance of each point from all the other points.
13. The algorithm is not suitable for large data sets as it requires more time to compute the distance of each point from all the other points.

## How to use the algorithm?

1. Import the KNeighborsClassifier class from the sklearn.neighbors library.
2. Create an instance of the KNeighborsClassifier class.
3. Train the model using the fit() method.
4. Predict the response for a new observation using the predict() method.
5. Done!

## How to improve the algorithm?

1. Normalize the data.
2. Use feature selection.
3. Use dimensionality reduction.
4. Use distance weighting.
5. Use cross-validation.
6. Use grid search.
7. Use ensemble methods.

## What are the advantages of the algorithm?

1. Simple to implement.
2. Simple to understand.
3. Simple to interpret.
4. Simple to visualize.
5. Simple to explain.
6. Simple to use.
7. Simple to tune.
8. Simple to debug.
9. Simple to extend.
10. Simple to modify.
11. Simple to combine with other algorithms.

## What are the disadvantages of the algorithm?

1. Computationally expensive.
2. High memory requirement.
3. High time requirement.
4. High space requirement.
5. High complexity.
6. High dimensionality.
7. High sparsity.
8. High noise.

## What are the applications of the algorithm?

1. Classification.
2. Regression.
3. Clustering.
4. Outlier detection.
5. Anomaly detection.
6. Density estimation.
7. Dimensionality reduction.
8. Feature selection.
9. Feature extraction.
10. Feature learning.

## STEP BY STEP IMPLEMENTATION OF KNN ALGORITHM

1. Import the required libraries.
2. Load the dataset.
3. Split the dataset into training and testing sets.
4. Create an instance of the KNeighborsClassifier class.
5. Train the model using the fit() method.
6. Predict the response for a new observation using the predict() method.
7. Evaluate the model using the accuracy_score() method.
8. Done!

```python
# Import the required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('data.csv')

# Split the dataset into training and testing sets
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
# Create an instance of the KNeighborsClassifier class
model = KNeighborsClassifier(n_neighbors=3)

# Train the model using the fit() method
model.fit(X_train, y_train)

# Predict the response for a new observation using the predict() method
y_pred = model.predict(X_test)

# Evaluate the model using the accuracy_score() method
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
```

## EXAMPLE OF KNN ALGORITHM

- STEP 1: Import the required libraries.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
```

- STEP 2: Load the dataset.

```python
data = pd.read_csv('data.csv')
```

- STEP 3: Split the dataset into training and testing sets.

```python
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- STEP 4: Create an instance of the KNeighborsClassifier class.

```python
model = KNeighborsClassifier(n_neighbors=3)
```

- STEP 5: Train the model using the fit() method.

```python
model.fit(X_train, y_train)
```

- STEP 6: Predict the response for a new observation using the predict() method.

```python
y_pred = model.predict(X_test)
```

- STEP 7: Evaluate the model using the accuracy_score() method.

```python
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
```