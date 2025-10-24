
# **Data Transformation in Machine Learning**

Machine learning datasets are often messy, inconsistent, and full of missing values. Training a model on such raw data leads to poor performance, so the data must be cleaned and converted into a usable format.

## **What is Data Transformation?**

Data transformation is the process of converting raw data into a cleaner, structured, and more suitable format for analysis and model training.

## **Why is it Important?**

Data transformation is a key step in the ML pipeline because it:

* Fixes issues like missing values, noise, and outliers
* Converts data into formats that models can understand
* Enables feature engineering for better predictions
* Improves data quality, leading to more accurate and reliable models

By transforming the data, machine learning algorithms can learn effectively and produce better results on unseen test data.

---

## **Different Data Transformation Technique**

Data transformation in machine learning involves a lot of techniques, let's discuss 8 of the major techniques that we can apply to data to better fit our model and produce better results in the prediction process.

The choice of data transformation technique depends on the characteristics of the data and the machine learning algorithm that we intend to use on the data. Here are the mentioned techniques discussed in details.

---

### **Handling Missing Data**

Datasets often contain missing values, which can lead to errors or poor model performance. Therefore, handling them is an essential step in data transformation. Two common approaches are:

* **Removing Missing Data:** Delete rows or columns with missing values when the missing portion is small. In pandas, this can be done using dropna(). However, removing too much data can reduce model accuracy.
* **Imputation:** Fill missing values instead of deleting them. Common methods include using the **mean, median, or mode**, a **constant value**, or advanced techniques like **KNNImputer** from sklearn.impute.

---

**Note:**
Before replacing missing values, certain factors must be carefully evaluated to ensure better results through imputation. These include:

* The *data type of the missing values*, which must be compatible with the chosen imputation method.
* The *distribution of the data*, since, for example, mean imputation is not suitable for skewed distributions.
* The selected imputation technique should *not distort the original variance or distribution* of the dataset.

---

### **Forward Fill and Backward Fill**

These methods are commonly used in **time-series analysis**, where data is recorded at regular intervals. When values are missing, they can be filled by:

* **Forward Fill (FFILL):** Replaces the missing value with the previous non-missing value.
* **Backward Fill (BFILL):** Replaces the missing value with the next non-missing value.

These techniques work well when the assumption holds that data points follow a natural temporal progression.

---

### **Interpolation**

Missing values can also be handled using **interpolation**, a method that predicts missing data based on existing values. The choice of interpolation technique depends on the nature of the dataset. One of the most commonly used approaches is **linear interpolation**, which assumes a linear relationship between adjacent observed values. In this method, a straight line is fitted between two known points, and the missing value is estimated accordingly.


### **Dealing with Outliers**

An **outlier** is a data point that is significantly different from the rest of the dataset. Outliers negatively affect the generalization of machine learning models as they can reduce performance and accuracy. Below are common techniques for handling outliers:

---

### **Identifying Outliers**

The first step in dealing with outliers is **identification**. This can be done using multiple methods:

* **Visual Inspection**:
  Use **box plots** or **scatter plots** to visually spot data points that lie far away from the majority of values.

* **Statistical Methods**:
  Techniques like the **Z-score** and **IQR (Interquartile Range)** can be used. For example, if the Z-score of a data point exceeds a chosen threshold, it can be flagged as an outlier.

* **Machine Learning Methods**:
  Models such as **Isolation Forest** and **One-Class SVM** can detect anomalies automatically.

> Choosing the right method depends on the nature of the data and the goal of the analysis.

---

### **Removing Outliers**

Outliers often act as noise or errors and removing them can improve model performance. However, removal should be done carefully. In cases like **fraud detection**, outliers may hold valuable insights and should be retained.

---

### **Transformations**

Outlier impact can be reduced using data transformation techniques such as:

* **Log Transformation** — useful for very large values.
* **Square Root Transformation** — milder than log transformation and suitable for positively skewed data.
* **Box-Cox Transformation** — helpful when the best transformation is not known in advance.

Choose the transformation based on the data distribution.

---

### **Truncation**

Truncation sets a threshold and adjusts all values outside that range. This reduces the effect of extreme values on model training and analysis.

---

### **Binning and Discretization**

Some algorithms (like decision trees) work better with discrete or categorical data. **Binning** converts continuous values into categories or bins. Tools like `KBinsDiscretizer` can be used to discretize continuous features and improve model performance.


---

Here is a **shorter, clearer, properly formatted Markdown version**. I also rewrote the formulas in **plain text format** so they are easy to read (instead of unreadable math blocks). No meaning has been changed.

---

### **Normalization and Standardization**

Normalization and Standardization are data transformation techniques used to **scale features to similar ranges**, helping machine learning models learn faster and perform better.

---

### **Normalization (Min–Max Scaling)**

* Scales values to a fixed range, usually **0 to 1**
* Useful when different features have very different ranges

**Formula (simple text):**

```
x_normalized = (x - min) / (max - min)
```

Here, `min` and `max` are the minimum and maximum values of the feature.

---

### **Standardization (Z-Score Scaling)**

* Transforms data so that **mean = 0** and **standard deviation = 1**
* Useful when data follows a **normal distribution**
* Helps gradient-based models converge faster

**Formula (simple text):**

```
x_standardized = (x - mean) / std
```

Here, `mean` is the feature’s average, and `std` is its standard deviation.

---

### **Summary**

| Method              | Output Range   | When to Use                    |
| ------------------- | -------------- | ------------------------------ |
| **Normalization**   | 0 to 1         | Features with different scales |
| **Standardization** | No fixed range | Data is normally distributed   |

---

### **Encoding Categorical Variables**

Many a times some features of a dataset are labeled as of different categories, but most of the machine learning algorithms works better on numeric data feature as compared to any different data type feature.

**One-Hot Encoding:**
One-Hot Encoding is the most common encoding techniques used in data transformation, what it does is that it converts each category in a categorical feature into a different binary feature(i.e. 0 or 1), for example if there is a feature called 'vehicle' in the dataset and the categories in it are 'car', 'bike', 'bicycle', one-hot encoding will create three separate columns as 'is_car', 'is_bike', 'is_bicycle' and then label them as 0 if absent or 1 if present.

**Label Encoding:**
Label Encoding on the other hand assigns a unique numeric value to different categories in the same feature, for example if there is a feature called size and it contains three values - 'small', 'medium', 'large', then the label encoding could label each value as 0, 1, 2 respectively.

**Ordinal Encoding:**
It is quite similar to label encoding except the fact that in ordinal encoding the categorical feature is encoded according to some sort of hierarchy in the system, For example if there are three categories in the categorical feature named - "High-School", "Bachelor's", and "Master's" the ordinal encoding will label this as 0, 1, 2 based on the educational hierarchy.

The choice of the encoding method depends on the nature of the categorical feature, the machine learning algorithm that we are using and the specific requirements of the given project.

---

### **Handling Skewed Distribution**

Many machine learning algorithms assumes that the data features are normally distributed, this is why handling skewed distribution becomes an essential task in data transformation process, as the skewed data might lead to biased or inaccurate model. As we have seen transformers in the Dealing with Outliers process of this article they are used usually to normally distribute the features, let's discuss some of the most common transformation techniques which will be used to handle skewed data:

**Logarithmic Transformation:**
This transformation technique is used when the data feature is right skewed or positively skewed. This transformation applies natural log values to all the data points, and thus it must be noticed that this technique only works on positive values only, as the log of positive values could only be taken.

**Square Root Transformation:**
Square root transformation is usually used in moderately right skewed data, it is less effective as compared to logarithmic transformation and it takes square root of each data point and makes it more like the data is normally distributed.

**Box-Cox Transformation**
Box-Cox Transformation is more suitable for right skewed data with data points either being positive or zero valued. It uses a parameter ( \lambda ) which helps in finding the best approximation to approximate the normality.

**Yeo-Johnson Transformation**
Yeo-Johnson transformation is more effective in nature, it works in a similar way as Box-Cox transformation, but it can work for both right as well as left skewed data feature, it can also work for either positive or negative value, which the box-cox transformation lacks.

**Qunatile Transformation**
This type of transformation works both for right skewed as well as left skewed data variable. Here the data is distributed according to there percentile in which they lay, hence, distributing the data into uniform sets.

The choice of transformation technique depends on the data we are working on and the skewness of the data, it is always preferred to visualize the data before applying the transformation technique.

---

### **Feature Engineering**

The process of creating new features or modifying the existing feature to improve the performance of machine learning model is called feature engineering. It helps in creating more informative and effective representation of patterns present in data by combining and transforming the given features. Through feature engineering we can increase our model performance and generalization ability. We have already seen some of the feature engineering techniques such as binning and normalization in previous steps, let's discuss some of the other most important techniques which we haven't discusses:

**Polynomial Features**
We can capture non linear relationships in the data by taking polynomial features into consideration, by squaring, cubing or increasing the power of the feature present. This technique can add flexibility to basically the linear models that are well known like linear regression, logistic regression etc. We can apply regularization with polynomial features to reduce the risk of overfitting.

**Interaction Terms**
We can combine two or more features together to produce a new feature, this helps machine learning algorithms specially linear models to identify and leverage the combined effect of different features on the outcome. The interaction terms uncovers the patterns that are not focused if individual features are considered, these terms helps in understanding the relationship between different variables and the effect of change in one feature on the behaviour of another. For example suppose we are modelling a simple regression problem of house price prediction, there are different house whose length and width of span is given let them be 'l', 'b' respectively. It is better to introduce a new feature area which is the multiplication on length and width, or 'l.b', which is a better indication of house price.

**Domain-Specific Feature**
We must consider creating features that are highly relevant and informative about the problem in hand. This process involves deep understanding of the domain on work we are in, as well as the knowledge of the data presented to us. This helps us create new features that might not be that much of use immediately but is essential for domain we are currently performing analysis in.

---

### **Dimensionality Reduction**

Dimentionality reduction is the process of reducing the number of features in the dataset while preserving the information that the original dataset conveys. It is often considered good to reduce the dimensions of highly dimensional dataset to reduce the computational complexity and reduce the chances of overfitting. There are two dimensionality reduction techniques which are used widely.

**Principal Component Analysis(PCA)**
PCA is the most common dimensionality reduction technique used in machine learning which transforms higher dimension data into lower dimension data retaining the information of the original dataset. PCA deals with the generation of principal components through standardization of data, finding covariance matrix of the data and then arranging the eigenvector obtained from the covariance matrix according to eigen values in descending order. In PCA the original data is projected onto the principal components to obtain lower dimensional data.

**t-Distributed Stochastic Neighbor Embedding**
t-SNE reduces the dimensionality of the data while maintaining local relationships between data points. What t-SNE algorithm does is that it takes higher dimensional data and finds out the similarities in between the data points, such that if this data point occurs what is the probability of the other data point occurring, and then it does the same with lower dimensional data and tries to reduce the divergence between the pairwise data points in high and low dimension space.

---

### **Text Data Transformation**

Text Data Transformation prepares the textual information for the machine learning models, usually raw text data is not suitable for machine learning algorithms, therefore, converting it into a suitable format becomes a part of the whole data transformation such that it could be fed into the machine learning algorithm. Let's discuss about some of the techniques used in text data transformation:

**Text Cleaning**
When we receive the textual data fromm any source of data, the raw data might contain HTML tags, punctuations, special characters, and symbols which is not usually useful in the analysis process, therefore, stripping them off of the data might be a better option. Also, converting the characters in the data into a lowercase is often considered a good practice, so that uniformity could be obtained in the data.

**Tokenization**
This process breaks down text into individual words or tokens, it is considered one of the most fundamental steps in word processing through nlp. The tokenization of raw text into structured format makes it easy to process and analyse words. For example - "This is a statement" could be tokenized as "This", "is", "a", "statement".

**Stopword Removal**
We must consider removing the words that do not contribute to the overall meaning of the text or the words which are not essential for the analysis process. For example words like - "and", "or", "the", etc.

**Stemming and Lemmatization**
Stemming is the reduction of words to their base forms like "sleeping" becomes associated with "sleep" or we can say that "sleep" is the base word here. Whereas, lemmatization is similar to stemming in many ways but it uses the core meaning of the word to get it to their base form. For example "worse" could be associated with "bad".

**TF-IDF**
Time Frequency - Inverse Document Frequency is the importance of a given word in a specified document with respect to the word's importance in a collection of documents. This process assigns higher weights to words which have more importance in a specific document as compared to different collection of documents.

**Word Embeddings**
This is the process of mapping out words as vector for higher dimensional space, such that similar words are mapped close to each other. Word embeddings helps in capturing the relationships between words.

---

### **Advantages and Disadvantages of Data Transformation**

There are several advantages of using data transformation, but with positive points there are also negative points that we must pay attention to such that we can achieve our goals that we have set from each project in hand. Let's discuss some of the advantages as well as disadvantages of data transformation that we must pay attention to such that we can best use our knowledge to improve our model's performance:

**Advantages**

* Improved Model Performance: The model gets better at generalizing new data when the issues in data are resolved through data transformation.
* Handling Missing Data: Results into good increase in the accuracy of the model.
* Better Convergence: Data normalization and standardization results into better convergence of the model during it's training period.
* Dimensionality Reduction: Simplifies the model training process.
* Better Insights from Feature Engineering

**Disadvantages:**

* Information Loss: If the transformation is not within limit valuable details might get discarded leading to a failed model.
* Risk of Overfitting: Excessive feature engineering and complex transformations might lead to overfitting, the model might fit data too closely and perform bad on new, unseen cases.
* Data Leakage: Applying transformation inappropriately using the entire dataset including the test data can lead to overestimation of model performance.
* Increased Complexity
* Assumption Violation: Sometimes transformation might not align with the assumptions of the chosen machine learning algorithm, which might lead to bad performance of the model in general.

Thus we can say that data transformation is a crucial step in machine learning, it requires careful consideration before deciding which techniques to apply to the data. While data transformation can improve model performance, it's important to avoid potential pitfalls such as information loss and overfitting. Cross-validation and evaluating model performance on unseen data are essential steps to ensure that the chosen data transformation techniques are appropriate and effective for the given task.

---

## **Conclusion**

In conclusion, data transformation is vital for refining raw data, improving model performance, and ensuring accurate machine learning outcomes.



