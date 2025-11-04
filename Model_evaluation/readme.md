
# Machine Learning Model Evaluation  


Model evaluation is a process that uses some metrics which help us to analyze the performance of the model. Think of training a model like teaching a student. Model evaluation is like giving them a test to see if they truly learned the subject—or just memorized answers. It helps us answer:

- Did the model learn patterns?  
- Will it fail on new questions?  

Model development is a multi-step process and we need to keep a check on how well the model do future predictions and analyze a models weaknesses. There are many metrics for that. **Cross Validation** is one technique that is followed during the training phase and it is a model evaluation technique.  

---

## Cross-Validation: The Ultimate Practice Test  

Cross Validation is a method in which we do not use the whole dataset for training. In this technique some part of the dataset is reserved for testing the model. There are many types of Cross-Validation out of which **K Fold Cross Validation** is mostly used.  

- In K Fold Cross Validation the original dataset is divided into k subsets. The subsets are known as folds.  
- This is repeated k times where 1 fold is used for testing purposes, rest k-1 folds are used for training the model.  
- This technique generalizes the model well and reduces the error rate.  

**Holdout** is the simplest approach. It is used in neural networks as well as in many classifiers. In this technique the dataset is divided into train and test datasets. The dataset is usually divided into ratios like **70:30** or **80:20**. Normally a large percentage of data is used for training the model and a small portion of dataset is used for testing the model.  

---

## Evaluation Metrics for Classification Task  

Classification is used to categorize data into predefined labels or classes. To evaluate the performance of a classification model we commonly use metrics such as **accuracy, precision, recall, F1 score and confusion matrix**.  

These metrics are useful in assessing how well model distinguishes between classes especially in cases of imbalanced datasets. By understanding the strengths and weaknesses of each metric, we can select the most appropriate one for a given classification problem.  

---

### Example: Decision Tree on Iris Dataset  

In this Python code, we have imported the iris dataset which has features like the length and width of sepals and petals. The target values are **Iris setosa, Iris virginica, and Iris versicolor**.  

After importing the dataset we divided the dataset into train and test datasets in the ratio 80:20. Then we called **Decision Trees** and trained our model. After that, we performed the prediction and calculated the accuracy score, precision, recall, and f1 score. We also plotted the confusion matrix.  

#### Importing Libraries and Dataset  

```python
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
````

Now let's load the toy dataset iris flowers from the `sklearn.datasets` library and then split it into training and testing parts (for model evaluation) in the 80:20 ratio.

```python
iris = load_iris()
X = iris.data
y = iris.target

# Holdout method.Dividing the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=20,
                                                    test_size=0.20)
```

Now, let's train a Decision Tree Classifier model on the training data, and then we will move on to the evaluation part of the model using different metrics.

```python
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
```

---

### 1. Accuracy

Accuracy is defined as the ratio of number of correct predictions to the total number of predictions.

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

However Accuracy has a drawback. It cannot perform well on an imbalanced dataset. Suppose a model classifies that the majority of the data belongs to the major class label. It gives higher accuracy, but in general model cannot classify on minor class labels and has poor performance.

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
```

**Output:**

```
Accuracy: 0.9333333333333333
```

---

### 2. Precision and Recall

* **Precision** is the ratio of true positives to the summation of true positives and false positives.

$$
Precision = \frac{TP}{TP + FP}
$$

* **Recall** is the ratio of true positives to the summation of true positives and false negatives.

$$
Recall = \frac{TP}{TP + FN}
$$

```python
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print('Recall:', recall_score(y_test, y_pred, average="weighted"))
```

**Output:**

```
Precision: 0.9435897435897436
Recall: 0.9333333333333333
```

---

### 3. F1 Score

F1 score is the harmonic mean of precision and recall.

$$
F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}
$$

```python
print('F1 score:', f1_score(y_test, y_pred, average="weighted"))
```

**Output:**

```
F1 score: 0.9327777777777778
```

---

### 4. Confusion Matrix

Confusion matrix is a N x N matrix where N is the number of target classes. It represents number of actual outputs and predicted outputs.

* True Positives (TP): actual = YES, predicted = YES
* True Negatives (TN): actual = NO, predicted = NO
* False Positives (FP): actual = NO, predicted = YES
* False Negatives (FN): actual = YES, predicted = NO

```python
from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix,
    display_labels=[0, 1, 2])

cm_display.plot()
plt.show()
```

**Output:** Confusion matrix for the model.

* Model accuracy = **93.33%**
* Precision ≈ **0.944**
* Recall = **0.933**
* F1 Score ≈ **0.933**

Class labels:

* 0 = Setosa
* 1 = Versicolor
* 2 = Virginica

From the confusion matrix, we see that:

* 8 Setosa classes were correctly predicted.
* 11 Versicolor test cases were correctly predicted.
* 2 Virginica cases were misclassified, rest 9 were correct.

---

### 5. AUC-ROC Curve

* **AUC (Area Under Curve)** evaluates classification model at different thresholds.
* **ROC (Receiver Operating Characteristic)** curve compares:

  * **TPR (Recall)** = TP / (TP+FN)
  * **FPR** = FP / (FP+TN)

```python
import numpy as np
from sklearn.metrics import roc_auc_score

y_true = [1, 0, 0, 1]
y_pred = [1, 0, 0.9, 0.2]
auc = np.round(roc_auc_score(y_true, y_pred), 3)
print("Auc", (auc))
```

**Output:**

```
Auc 0.75
```

A model is considered **good** if the AUC score is **> 0.5** and approaches **1**.

---

## Evaluation Metrics for Regression Task

Regression is used to determine continuous values. It is mostly used to find a relation between a dependent and independent variable.

Unlike classification (accuracy, F1, confusion matrix), regression uses **error-based metrics**.

---

### Example: Linear Regression on Mumbai Weather Dataset

We use the dataset: **Day, Hour, Temperature, Relative Humidity, Wind Speed, Wind Direction.**

* Dependent Variable = **Relative Humidity**
* Independent Variable = **Temperature**

#### Import Libraries

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
```

#### Load Data

```python
df = pd.read_csv('weather.csv')
X = df.iloc[:, 2].values
Y = df.iloc[:, 3].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.20,
                                                    random_state=0)
```

#### Train Model

```python
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

regression = LinearRegression()
regression.fit(X_train, Y_train)
Y_pred = regression.predict(X_test)
```

---

### 1. Mean Absolute Error (MAE)

$$
MAE = \frac{\sum |y_{pred} - y_{actual}|}{N}
$$

```python
mae = mean_absolute_error(y_true=Y_test, y_pred=Y_pred)
print("Mean Absolute Error", mae)
```

**Output:**

```
Mean Absolute Error 1.7236295632503873
```

---

### 2. Mean Squared Error (MSE)

$$
MSE = \frac{\sum (y_{pred} - y_{actual})^2}{N}
$$

```python
mse = mean_squared_error(y_true=Y_test, y_pred=Y_pred)
print("Mean Square Error", mse)
```

**Output:**

```
Mean Square Error 3.9808057060106954
```

---

### 3. Root Mean Squared Error (RMSE)

$$
RMSE = \sqrt{\frac{\sum (y_{pred} - y_{actual})^2}{N}}
$$

```python
rmse = mean_squared_error(y_true=Y_test, y_pred=Y_pred, squared=False)
print("Root Mean Square Error", rmse)
```

**Output:**

```
Root Mean Square Error 1.9951956560725306
```

---

### 4. Mean Absolute Percentage Error (MAPE)

$$
MAPE = \frac{1}{N} \sum \left(\frac{|y_{pred} - y_{actual}|}{|y_{actual}|}\right) \times 100
$$

```python
mape = mean_absolute_percentage_error(Y_test, Y_pred,
                                      sample_weight=None,
                                      multioutput='uniform_average')
print("Mean Absolute Percentage Error", mape)
```

**Output:**

```
Mean Absolute Percentage Error 0.02334408993333347
```

---

## Conclusion

Evaluating machine learning models is an important step in ensuring their effectiveness and reliability in real-world applications.

* **Classification Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix, AUC-ROC
* **Regression Metrics**: MAE, MSE, RMSE, MAPE
* **Evaluation Techniques**: Cross-validation, Holdout

Using the right metric ensures that models **generalize well** and are **robust for unseen data**.


