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

