# Default Parameters of Common Machine Learning Models

This document lists the default (and explicitly defined) parameters for commonly used models in scikit-learn.

---

##  Logistic Regression
```python
LogisticRegression(random_state=42)
```
| Parameter | Description | Default |
|------------|--------------|----------|
| penalty | Type of regularization (‘l1’, ‘l2’, ‘elasticnet’, or ‘none’) | 'l2' |
| dual | Dual or primal formulation | False |
| tol | Tolerance for stopping criteria | 1e-4 |
| C | Inverse of regularization strength | 1.0 |
| fit_intercept | Whether to add an intercept term | True |
| solver | Algorithm used in optimization | 'lbfgs' |
| max_iter | Maximum number of iterations | 100 |
| multi_class | Multi-class handling | 'auto' |
| random_state | For reproducibility | 42 |

---

## Support Vector Machine (SVM)
```python
SVC(random_state=42)
```
| Parameter | Description | Default |
|------------|--------------|----------|
| C | Regularization parameter | 1.0 |
| kernel | Kernel type (‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, etc.) | 'rbf' |
| degree | Degree of polynomial kernel | 3 |
| gamma | Kernel coefficient | 'scale' |
| coef0 | Independent term in kernel function | 0.0 |
| probability | Enable probability estimates | False |
| shrinking | Use the shrinking heuristic | True |
| random_state | For reproducibility | 42 |

---

##  Random Forest Classifier
```python
RandomForestClassifier(random_state=42)
```
| Parameter | Description | Default |
|------------|--------------|----------|
| n_estimators | Number of trees in the forest | 100 |
| criterion | Function to measure split quality | 'gini' |
| max_depth | Maximum depth of each tree | None |
| min_samples_split | Minimum samples to split an internal node | 2 |
| min_samples_leaf | Minimum samples required at a leaf node | 1 |
| max_features | Number of features considered for best split | 'sqrt' |
| bootstrap | Whether bootstrap samples are used | True |
| random_state | For reproducibility | 42 |

---

##  Decision Tree Classifier
```python
DecisionTreeClassifier(random_state=42)
```
| Parameter | Description | Default |
|------------|--------------|----------|
| criterion | Measure for split quality | 'gini' |
| splitter | Strategy to choose split | 'best' |
| max_depth | Maximum depth of tree | None |
| min_samples_split | Minimum samples to split | 2 |
| min_samples_leaf | Minimum samples at a leaf node | 1 |
| max_features | Number of features considered per split | None |
| random_state | For reproducibility | 42 |

---

##  K-Nearest Neighbors (KNN)
```python
KNeighborsClassifier()
```
| Parameter | Description | Default |
|------------|--------------|----------|
| n_neighbors | Number of nearest neighbors | 5 |
| weights | Weight function (‘uniform’ or ‘distance’) | 'uniform' |
| algorithm | Algorithm for nearest neighbors | 'auto' |
| leaf_size | Leaf size passed to BallTree/KDTree | 30 |
| p | Power parameter for Minkowski distance (p=2 → Euclidean) | 2 |
| metric | Distance metric | 'minkowski' |

---

