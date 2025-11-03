# AUC ROC Curve in Machine Learning
AUC-ROC curve is a graph used to check how well a binary classification model works. It helps us to understand how well the model separates the positive cases like people with a disease from the negative cases like people without the disease at different threshold level. It shows how good the model is at telling the difference between the two classes by plotting:

  1. **True Positive Rate (TPR)**: how often the model correctly predicts the positive cases also known as Sensitivity or Recall.
  2. **False Positive Rate (FPR)**: how often the model incorrectly predicts a negative case as positive.
  3. **Specificity**: measures the proportion of actual negatives that the model correctly identifies. It is calculated as 1 - FPR.
   
   The higher the curve the better the model is at making correct predictions.
   
<img width="450" height="325" alt="image" src="https://github.com/user-attachments/assets/8dcd75dd-9a87-4fbe-a8a9-ddaa4772217e" />

These terms are derived from the confusion matrix which provides the following values:

**True Positive (TP)**: Correctly predicted positive instances

**True Negative (TN)**: Correctly predicted negative instances

**False Positive (FP)**: Incorrectly predicted as positive

**False Negative (FN)**: Incorrectly predicted as negative

<img width="500" height="152" alt="image" src="https://github.com/user-attachments/assets/92e08b67-bb5d-4868-a3e2-16060548a2d5" />

**ROC Curve :** It plots TPR vs. FPR at different thresholds. It represents the trade-off between the sensitivity and specificity of a classifier.

**AUC(Area Under the Curve):** measures the area under the ROC curve. A higher AUC value indicates better model performance as it suggests a greater ability to distinguish between classes. An AUC value of 1.0 indicates perfect performance while 0.5 suggests it is random guessing.


**AUC-ROC curve helps us understand how well a classification model distinguishes between the two classes. Imagine we have 6 data points and out of these**


**3 belong to the positive class:** Class 1 for people who have a disease.

**3 belong to the negative class:** Class 0 for people who don’t have disease.

<img width="390" height="217" alt="image" src="https://github.com/user-attachments/assets/62a72019-9556-4676-974c-8a7d49028730" />

Now the model will give each data point a predicted probability of belonging to Class 1. The AUC measures the model's ability to assign higher predicted probabilities to the positive class than to the negative class. Here’s how it work:

**1. Randomly choose a pair:** Pick one data point from the positive class (Class 1) and one from the negative class (Class 0).

**2. Check if the positive point has a higher predicted probability:** If the model assigns a higher probability to the positive data point than to the negative one for correct ranking.

**3. Repeat for all pairs:** We do this for all possible pairs of positive and negative examples.

## When to Use AUC-ROC

AUC-ROC is effective when:

  The dataset is balanced and the model needs to be evaluated across all thresholds.
  
  False positives and false negatives are of similar importance.
  
  In cases of highly imbalanced datasets AUC-ROC might give overly optimistic results. In such cases the Precision-Recall Curve is more suitable focusing on the positive class.

## Model Performance with AUC-ROC:

**High AUC (close to 1):** The model effectively distinguishes between positive and negative instances.

**Low AUC (close to 0):** The model struggles to differentiate between the two classes.

**AUC around 0.5:** The model doesn’t learn any meaningful patterns i.e it is doing random guessing.

**In short AUC gives you an overall idea of how well your model is doing at sorting positives and negatives, without being affected by the threshold you set for classification. A higher AUC means your model is doing good.**
