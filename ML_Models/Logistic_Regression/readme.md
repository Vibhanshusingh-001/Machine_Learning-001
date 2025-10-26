
# Logistic Regression – Detailed Explanation

## 1. Introduction

Logistic Regression is a **supervised learning algorithm** used for:

* **Binary Classification:** Predicting 0 or 1 (e.g., spam vs. non-spam, disease vs. healthy)
* **Multiclass Classification:** Extending binary logistic regression to multiple classes (via softmax or OvR)

Although it has “regression” in its name, it is primarily used for **classification**, not predicting continuous values. It models the **probability** of a class using the **logistic (sigmoid) function**.

---

## 2. Key Concepts

* **Linear Model:** Logistic regression computes a linear combination of features:

  ```
  z = β0 + β1*x1 + β2*x2 + ... + βn*xn
  ```

* **Sigmoid Function:** Converts linear output to probability (between 0 and 1):

  ```
  sigmoid(z) = 1 / (1 + e^(-z))
  ```

* **Decision Rule:**

  ```
  if sigmoid(z) >= 0.5 -> predict 1
  else -> predict 0
  ```

* **Interpretation of Coefficients:**

  * βi represents the **log-odds** change for a unit increase in feature xi.

---

## 3. Odds and Log-Odds

* **Odds:**

  ```
  Odds = p / (1 - p)
  ```

* **Log-Odds (Logit):**

  ```
  logit(p) = log(p / (1 - p)) = β0 + β1*x1 + ... + βn*xn
  ```

* Logistic regression models the **log-odds** as a linear function of input features.

---

## 4. Cost Function

* **Binary Cross-Entropy / Log Loss:**

  ```
  J(β) = -(1/n) * Σ [ yi*log(y_hat_i) + (1 - yi)*log(1 - y_hat_i) ]
  ```

  where:

  * yi = true label
  * y_hat_i = predicted probability

* Goal: Minimize log loss using optimization algorithms like gradient descent or liblinear solver.

---

## 5. Multiclass Logistic Regression

* **One-vs-Rest (OvR):** Fit one binary classifier per class.

* **Softmax / Multinomial:** Generalization of sigmoid to multiple classes:

  ```
  P(y=k | x) = exp(βk . x) / Σ(exp(βj . x))  for j=1 to K
  ```

* Use `multi_class='ovr'` or `multi_class='multinomial'` in Scikit-Learn.

---

## 6. Regularization

Regularization prevents overfitting:

* **L2 Regularization (Ridge):**

  ```
  Cost = J(β) + (λ/2) * Σ(βj^2)
  ```

* **L1 Regularization (Lasso):**

  ```
  Cost = J(β) + λ * Σ |βj|
  ```

* **Elastic Net:** Combination of L1 and L2 regularization.

* **C parameter in Scikit-Learn:**

  ```
  C = 1 / λ
  ```

  Larger `C` → less regularization; smaller `C` → stronger regularization.

---

## 7. Logistic Regression Parameters (Scikit-Learn)

* `penalty` → Type of regularization: `'l1'`, `'l2'`, `'elasticnet'`, `'none'`
* `C` → Inverse of regularization strength
* `solver` → Optimization algorithm: `'liblinear'`, `'lbfgs'`, `'saga'`, `'newton-cg'`, `'sag'`
* `max_iter` → Maximum iterations for convergence
* `multi_class` → `'ovr'` or `'multinomial'`
* `fit_intercept` → Include intercept (bias) term
* `intercept_scaling` → Scaling for intercept (liblinear only)
* `class_weight` → Weights for imbalanced classes
* `random_state` → Seed for reproducibility
* `tol` → Tolerance for stopping criterion
* `l1_ratio` → Elastic Net mixing parameter (0 = L2, 1 = L1)

---

## 8. Advantages and Limitations

**Advantages:**

* Simple and interpretable
* Outputs probabilities
* Works well for linearly separable data
* Can handle binary and multiclass classification

**Limitations:**

* Assumes linear relationship between log-odds and features
* Not suitable for complex non-linear boundaries
* Sensitive to multicollinearity
* Performance may drop with high-dimensional sparse data

---

## 9. Hyperparameter Tuning

Key parameters to tune:

* `C` → Regularization strength
* `penalty` → L1, L2, or Elastic Net
* `solver` → Depends on penalty and dataset size
* `multi_class` → OvR or multinomial
* `max_iter` → Ensure convergence

Example:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200]
}

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_)
```

---

## 10. Full Working Example

### Binary Classification

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

### Multiclass Classification

```python
model = LogisticRegression(C=1.0, multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## 11. Summary

* Logistic Regression models **probability of a class** using the **sigmoid function**
* Linear combination of features → log-odds → probability
* Supports **regularization** to prevent overfitting
* Key hyperparameters: `C`, `penalty`, `solver`, `multi_class`, `max_iter`
* Simple, interpretable, and widely used for classification problems

