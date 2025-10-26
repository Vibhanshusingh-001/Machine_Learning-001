
# Support Vector Machine (SVM) 

## 1. Introduction

Support Vector Machine (SVM) is a **supervised machine learning algorithm** used for:

* **Classification:** Predicting discrete labels (e.g., spam vs. non-spam, disease vs. healthy)
* **Regression (SVR):** Predicting continuous values

SVM works by finding a **hyperplane** that best separates data points of different classes while maximizing the margin between them. It is particularly effective in **high-dimensional spaces** and with **non-linear boundaries** using kernel functions.

---

## 2. Key Concepts

* **Hyperplane:** A decision boundary that separates data into classes.

  * In 2D: line
  * In 3D: plane
  * In higher dimensions: hyperplane

* **Margin:** The distance between the hyperplane and the closest data points from each class.

  * Goal of SVM: **maximize the margin** → ensures better generalization

* **Maximum Margin Classifier:** The hyperplane that maximizes the margin between classes.

---

## 3. Support Vectors

* **Support vectors** are the data points **closest to the hyperplane**.
* Only these points determine the hyperplane; other points have no direct impact.
* Removing non-support vector points does not change the decision boundary.

---

## 4. Hard Margin vs Soft Margin

* **Hard Margin SVM:**

  * No misclassifications allowed
  * Only works if data is perfectly linearly separable
  * Prone to overfitting in noisy data

* **Soft Margin SVM:**

  * Allows some misclassifications
  * Introduces a **penalty parameter C** to balance margin maximization and misclassification
  * More practical for real-world data

---

## 5. Hinge Loss & The C Parameter

* **Hinge Loss Function:**
  Used to penalize misclassified points:
  [
  L = \sum_{i=1}^{n} \max(0, 1 - y_i (\mathbf{w} \cdot \mathbf{x_i} + b))
  ]

* **C (Regularization Parameter):**

  * High C → less tolerance for misclassification (hard margin-like)
  * Low C → allows more misclassifications, wider margin (soft margin)
  * Balances **bias vs variance** trade-off

---

## 6. Kernel Trick

SVM can separate **non-linear data** by mapping it into **higher-dimensional space** where it becomes linearly separable.

* **Kernel Function:** Computes the inner product in higher-dimensional space without explicitly mapping data.
* Common kernels:

  * **Linear Kernel**: No mapping, best for linearly separable data
  * **Polynomial Kernel**: Maps to polynomial features
  * **RBF (Gaussian) Kernel**: Non-linear, widely used
  * **Sigmoid Kernel**: Similar to neural networks

---

## 7. Types of Kernels (Detailed)

| Kernel         | Function                                    | Use-case / Notes                              |        |   |       |                                       |
| -------------- | ------------------------------------------- | --------------------------------------------- | ------ | - | ----- | ------------------------------------- |
| Linear         | ( K(x, x') = x \cdot x' )                   | Linearly separable data; simplest and fastest |        |   |       |                                       |
| Polynomial     | ( K(x, x') = (\gamma x \cdot x' + r)^d )    | Captures polynomial relationships; d = degree |        |   |       |                                       |
| RBF (Gaussian) | ( K(x, x') = \exp(-\gamma                   |                                               | x - x' |   | ^2) ) | Most popular; handles non-linear data |
| Sigmoid        | ( K(x, x') = \tanh(\gamma x \cdot x' + r) ) | Rarely used; behaves like a neural network    |        |   |       |                                       |

**Kernel Parameters:**

* **gamma (γ):** Controls influence of a single training point. Low γ → far influence, high γ → close influence
* **degree:** Only for polynomial kernel; degree of polynomial
* **coef0:** Only for polynomial & sigmoid; trade-off parameter between higher-order and lower-order terms

---

## 8. How SVM Works on Non-linear Data

1. Original data is non-linearly separable in input space
2. Kernel function maps data to a higher-dimensional feature space
3. Linear hyperplane is computed in this new space
4. Decision boundary in original space becomes non-linear

---

## 9. SVM Parameters (Complete)

### Classification (`SVC`)

| Parameter                 | Description                                                                               |
| ------------------------- | ----------------------------------------------------------------------------------------- |
| `C`                       | Regularization parameter (high → less tolerance for misclassification)                    |
| `kernel`                  | Kernel type: `linear`, `poly`, `rbf`, `sigmoid`, `precomputed`                            |
| `degree`                  | Degree of polynomial kernel (default 3)                                                   |
| `gamma`                   | Kernel coefficient for `rbf`, `poly`, `sigmoid`. Options: `scale`, `auto`, or float value |
| `coef0`                   | Independent term in kernel function for `poly` and `sigmoid`                              |
| `shrinking`               | Whether to use shrinking heuristics (True/False)                                          |
| `probability`             | Enable probability estimates (slower)                                                     |
| `tol`                     | Tolerance for stopping criterion                                                          |
| `cache_size`              | Size of kernel cache (MB)                                                                 |
| `class_weight`            | Weights associated with classes, useful for imbalanced data                               |
| `verbose`                 | Enable verbose output (True/False)                                                        |
| `max_iter`                | Maximum number of iterations (-1 = no limit)                                              |
| `decision_function_shape` | ‘ovo’ (one-vs-one) or ‘ovr’ (one-vs-rest)                                                 |
| `random_state`            | Seed for reproducibility (used when probability=True)                                     |

### Regression (`SVR`)

Additional parameters:

| Parameter                                                                  | Description                                             |
| -------------------------------------------------------------------------- | ------------------------------------------------------- |
| `epsilon`                                                                  | Margin of tolerance where no penalty is given to errors |
| `C`                                                                        | Regularization parameter (high → less tolerance)        |
| `kernel`                                                                   | Kernel type (`linear`, `poly`, `rbf`, `sigmoid`)        |
| `gamma`                                                                    | Kernel coefficient                                      |
| `degree`, `coef0`, `shrinking`, `tol`, `cache_size`, `verbose`, `max_iter` | Same as SVC                                             |

### Linear SVM (`LinearSVC`, `LinearSVR`)

* Faster for large datasets
* Uses **liblinear solver**
* Does not support probability estimates directly
* Parameters: `C`, `loss` (hinge or squared_hinge), `penalty` (l1/l2), `dual` (True/False), `tol`, `max_iter`, `class_weight`, `random_state`

---

## 10. Advantages and Limitations

**Advantages:**

* Effective in high-dimensional spaces
* Works well with both linear and non-linear data
* Memory efficient (uses only support vectors)
* Robust to overfitting with proper regularization

**Limitations:**

* Choosing the right kernel can be tricky
* Training can be slow on very large datasets
* Less interpretable compared to decision trees
* Sensitive to feature scaling

---

## 11. Hyperparameter Tuning

Key parameters to tune:

* **C:** Regularization
* **gamma:** Kernel coefficient (for `rbf`, `poly`, `sigmoid`)
* **kernel:** Linear, RBF, Polynomial
* **degree:** For polynomial kernel
* **coef0:** For polynomial and sigmoid kernels

Example using GridSearchCV:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}

grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_)
```

---

## 12. Full Working Example

### Classification

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

model = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

### Regression (SVR)

```python
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

model = SVR(C=1.0, kernel='rbf', gamma='scale', epsilon=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
```

---

## 13. Summary

* SVM finds a hyperplane that maximizes the margin between classes
* Uses **support vectors** to define the decision boundary
* Can handle **non-linear data** using the **kernel trick**
* Key hyperparameters: `C`, `kernel`, `gamma`, `degree`, `coef0`
* Works well for high-dimensional datasets, but kernel selection and scaling are critical

SVM is a **robust, versatile, and powerful algorithm** for both classification and regression tasks when used with appropriate kernels and parameter tuning.


