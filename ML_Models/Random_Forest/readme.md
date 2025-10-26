

# Random Forest:

### 1. What is Random Forest?

Random Forest is a supervised ensemble learning algorithm that builds multiple decision trees and combines their outputs to make a final prediction. It works for:

* Classification problems
* Regression problems

It is based on two core ideas:

1. **Bagging (Bootstrap Aggregation):** Reduces variance and improves stability.
2. **Random Feature Selection:** Ensures trees are decorrelated and reduces overfitting.

Random Forest is widely used because it is robust, accurate, and performs well with minimal preprocessing.

---

### 2. How Random Forest Works (Step-by-Step)

Assume we want to build a Random Forest with 100 trees.

| Step                             | Description                                                         |
| -------------------------------- | ------------------------------------------------------------------- |
| Step 1: Bootstrap Sampling       | Create random subsets of the data (with replacement) for each tree. |
| Step 2: Train Decision Trees     | Train each tree on its bootstrap sample.                            |
| Step 3: Random Feature Selection | At each node split, only a random subset of features is considered. |
| Step 4: Prediction               | Classification: majority voting. Regression: average of outputs.    |
| Step 5: Final Result             | Ensemble reduces variance and improves accuracy.                    |

Random feature selection makes each tree unique, and averaging their results makes the overall model more stable.

---

### 3. Bias–Variance Intuition

* A single decision tree usually has **low bias and high variance**, making it prone to overfitting.
* Random Forest reduces variance by averaging many trees, without increasing bias significantly.

Result: **Low bias + low variance**, which leads to better generalization performance.

---

### 4. Advantages and Limitations

**Advantages:**

* Works well with most datasets
* Handles missing values and outliers
* Requires little to no feature scaling
* Less prone to overfitting than a single decision tree
* Can handle high-dimensional data

**Limitations:**

* Slower to train when the number of trees is large
* Uses more memory
* Less interpretable than a single decision tree

---

### 5. Important Parameters in RandomForest (Scikit-Learn)

| Parameter           | Description                                                                                                                         |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `n_estimators`      | Number of trees in the forest. Default: 100. More trees improve accuracy but increase training time.                                |
| `criterion`         | Function to measure split quality. Classification: `gini`, `entropy`, or `log_loss`. Regression: `squared_error`, `absolute_error`. |
| `max_depth`         | Maximum depth of each tree. Prevents overfitting. Default: `None` (tree grows fully).                                               |
| `min_samples_split` | Minimum samples required to split a node. Higher values reduce overfitting. Default: 2.                                             |
| `min_samples_leaf`  | Minimum samples required at a leaf node. Typical values: 1 to 4.                                                                    |
| `max_features`      | Number of features to consider at each split. Default for classification: `sqrt`. Default for regression: `log2` or `auto`.         |
| `bootstrap`         | Whether to use bootstrap samples. Default: `True`.                                                                                  |
| `oob_score`         | Use Out-of-Bag samples for validation. Default: `False`. Set to `True` to estimate performance without a test set.                  |
| `n_jobs`            | Number of CPU cores to use. `-1` means use all cores.                                                                               |
| `random_state`      | Controls randomness and ensures reproducibility.                                                                                    |
| `max_samples`       | If `bootstrap=True`, controls sample size per tree.                                                                                 |

---

### 6. How Prediction Works

| Task           | Method                                                             |
| -------------- | ------------------------------------------------------------------ |
| Classification | Each tree predicts a class; the final result is the majority vote. |
| Regression     | Each tree predicts a value; the final result is the average.       |

This approach is called voting (classification) and averaging (regression).

---

### 7. Out-of-Bag (OOB) Score

Since bootstrap sampling leaves out about 33 percent of the data for each tree, those unused samples are called **Out-of-Bag samples**. These samples are used as an internal validation set to estimate model performance.

Benefit: You get a validation score without a separate test split.

---

### 8. Hyperparameter Tuning

Most important parameters to tune:

| Parameter                                  | Reason                                        |
| ------------------------------------------ | --------------------------------------------- |
| `n_estimators`                             | Improves accuracy but increases training time |
| `max_depth`                                | Controls overfitting                          |
| `min_samples_split` and `min_samples_leaf` | Add regularization                            |
| `max_features`                             | Controls randomness and tree diversity        |

Example code using GridSearchCV:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
grid.fit(X, y)
print(grid.best_params_)
```

---

### 9. Full Working Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

### 10. Feature Importance

```python
import pandas as pd

importance = model.feature_importances_
feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
print(feat_imp.sort_values(by='Importance', ascending=False))
```

