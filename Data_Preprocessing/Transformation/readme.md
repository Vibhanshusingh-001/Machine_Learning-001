# Data Transformation in Machine Learning
Often the data received in a machine learning project is messy and missing a bunch of values, creating a problem while we try to train our model on the data without altering it.

In building a machine learning project that could predict the outcome of data well, the model requires data to be presented to it in a specific way such that the machine learning model could easily train on the data and perform prediction on the test dataset. In this article, we will be going through the nitty-gritty of data transformation in machine learning.

## What is Data Transformation?
Data transformation is the process of converting raw data into a more suitable format or structure for analysis, to improve its quality and make it compatible with the requirements of a particular task or system.

## Data Transformation in Machine Learning
Data transformation is the most important step in a machine learning pipeline which includes modifying the raw data and converting it into a better format so that it can be more suitable for analysis and model training purposes. In data transformation, we usually deal with issues such as noise, missing values, outliers, and non-normality.

## Why Data Transformation is Important?
Data transformation is crucial in the data analysis and machine learning pipeline as it plays an important role in preparing raw data for meaningful insights and accurate model building. Raw data, often sourced from diverse channels, may be inconsistent, contain missing values, or exhibit variations that could impact the reliability of analyses.

Data transformation addresses these challenges by cleaning, encoding, and structuring the data in a manner that makes it compatible with analytical tools and algorithms.

Additionally, data transformation facilitates feature engineering, allowing the creation of new variables that may improve model performance. By converting data into a more suitable format, ensures that models are trained on high-quality, standardized data, leading to more reliable predictions and valuable insights.

## Different Data Transformation Technique
Data transformation in machine learning involves a lot of techniques, let's discuss 8 of the major techniques that we can apply to data to better fit our model and produce better results in the prediction process.

The choice of data transformation technique depends on the characteristics of the data and the machine learning algorithm that we intend to use on the data. Here are the mentioned techniques discussed in details.

---

### Handling Missing Data
Most of the times the data that is received from different sources miss some of the values in it, if we train our model on this data the model might behave differently or even produce error while training. Therefore, handling missing data becomes an important aspect to consider while transforming the data, there are different techniques through which we can handle the missing data which can help us improve our model performance. Let's discuss some of the techniques in details here:

**Removing the Missing Data:** We can delete the rows or columns which are having missing data. This is significant only when a small number of data is missing, if there's a large value of missing data points in our dataset we must consider some other technique otherwise deleting the rows or columns with large number of missing values will change the way our model performs since it might cause the model to train on less data. We can drop the rows which contain the missing values using the dropna method in pandas if the data we have is stored in a pandas dataframe.

**Imputation:** In this technique we remove the missing values by filling the missing values positions with some other value, for example we can fill in the missing values with the mean of the different values from the same column which is in the same category or data type as of the missing value. The most common types of imputation methods include mean, median, mode imputation. We can also fill in the missing values with a constant value that we want to be present in the data instead of the missing value. Imputation is also implanted within the sklearn library, we can impute different missing values with the help of KNNImputer(K-Nearest Neighbors) which is a part of sklearn.impute.

**Note:**  
Before replacing the missing values we must consider some things in order to get better results in handling missing values through imputation method which includes the data type of the missing values must match imputation method, data distribution must also be considered for example mean imputation is not good for skewed data, the imputation method must not affect the variance and distribution of the original data present in the dataset.

**Forward Fill or Backward Fill:** Usually in time series analysis, where the data is produced after a constant time, if some data goes missing we can replace the missing value with forward fill or the backward fill options, The forward fill method fills the missing value with the previous non missing value whereas backward fill method fills the missing value with the next non missing value to the missing value.

**Interpolation:** Missing data can also be handled by interpolation technique, it involves predicting the missing values based on observed values in the dataset. There are multiple interpolation methods, the choice of the method is based on the data that we have. Most commonly used interpolation is linear interpolation which assumes there is a linear relationship between observed values and missing data points, this method predicts the missing value by fitting a straight line between two adjacent non-missing points.

---

### Dealing with Outliers
Dealing with the outlier values is one of the most important steps of data transformation, an outlier is a data point which is significantly different in value from the rest of the dataset. These outliers affects the generalization behaviour of the machine learning model as they impact the performance and accuracy of the model, here are some of the techniques used to handle the outliers:

**Identification of Outlier:** The first step to consider in dealing with outliers is to identify the outliers data points. This could be done through more than one method, the first way we can identify the outliers is through visual inspection we can use either box-plot or scatter plot to identify the data points which are not among most of the data points in the dataset, we can also obtain the outlier data points by statistical methods including z-score and IQR, for example, we can calculate the z-score for all the data points and consider a threshold, if the z-score of the data point crosses the certain mentioned threshold then it is considered to be an outlier. There are also certain machine learning anomaly detection models including Isolation Forest and One-Class SVM which could be used to identify the outliers. Choosing the correct outlier detection method depends on the characteristics of data and the goals of analysis.

**Removing Outliers:** The removal of outliers becomes an important measure in most of the cases as it produces noise or acts as an error and reduces the model performance. But before removing we must consider the characteristics of data which we are trying to analyze, for example in cases such as fraud detection the outlier can give important insights regarding faulty transactions.

**Transformations:** Through data transformation we can reduce the impact of outliers with the help of various data transformation methods like log transformation in which if the outlier value is really large this method can reduce the impact of the value, or we can use square root transformation as well which is suitable for positively skewed data like log transformation, but this method is quite milder than log transformation. We can also use Box-Cox transformation, where we are not sure which transformation is useful. There are many other methods of data transformation which we can choose but choosing the best according to the characteristics of data that we have is a good way to work on a project.

**Truncation:** Truncation is the method of setting a threshold and then adjusting all the points that are outside the range of the threshold value. Truncation reduces the impact of outliers on the analysis and modelling process by restricting the impact of extreme values.

**Binning and Discretization of Data:** Often times certain machine learning algorithms like decision tree perform better on categorical data, but the data we might receive from different sources can be continuous in value. Therefore converting the continuous range of values into bins of data could help improve model performance. Binning is the process of converting continuous value of a feature into discrete values, KBinsDiscretizer is one of the most commonly used discretization technique that could be applied on continuous value data to encode them into discrete values.

The technique of dealing with outlier must be chosen according to the characteristics of the data present in-front of us and the machine learning algorithm that we are applying on the data.

---

### Normalization and Standardization
Normalization and Standardization are two of the most common techniques used in data transformation which aims to scale and transform the data such that the features have similar scales, which makes it easy for the machine learning algorithm ot learn and converge.

**Normalization:**  
The main objective of normalization is to rescale the features to a standard range of values which is usually 0-1. Normalization is usually used when different features have different range of values and some feature might contribute more to the model learning process, normalization helps in equalizing the range of the features and makes sure that the features contribute equally to the learning algorithm.  
Mathematically after normalization the new data point becomes:  
\[
x_i' = \frac{x_i - min(X)}{max(X) - min(X)}
\]  
here max(X) and min(X) are the maximum and minimum value of the feature of the data point being used \(x_i\) is the value of the data point.

**Standardization:**  
Standardardization is also known as z-score normalization, the objective of standardization is to transform the feature such that the value of mean becomes 0 and the value of standard deviation becomes 1. Standardization is usually useful when features have different scales but follow normal distribution, it helps machine learning algorithms which relies on gradient based optimization to converge at a faster rate. The new data point after standardization becomes:  
\[
x_i' = \frac{x_i - mean(X)}{std(X)}
\]  
here, mean(X) and std(X) are the mean and standard deviation of feature respectively.

---

### Encoding Categorical Variables
Many a times some features of a dataset are labeled as of different categories, but most of the machine learning algorithms works better on numeric data feature as compared to any different data type feature. Therefore, encoding of categorical features becomes an important step of data transformation. The categorical features could be encoded into numerical valued features in different ways, let's discuss some of the most common encoding techniques:

**One-Hot Encoding:**  
One-Hot Encoding is the most common encoding techniques used in data transformation, what it does is that it converts each category in a categorical feature into a different binary feature(i.e. 0 or 1), for example if there is a feature called 'vehicle' in the dataset and the categories in it are 'car', 'bike', 'bicycle', one-hot encoding will create three separate columns as 'is_car', 'is_bike', 'is_bicycle' and then label them as 0 if absent or 1 if present.

**Label Encoding:**  
Label Encoding on the other hand assigns a unique numeric value to different categories in the same feature, for example if there is a feature called size and it contains three values - 'small', 'medium', 'large', then the label encoding could label each value as 0, 1, 2 respectively.

**Ordinal Encoding:**  
It is quite similar to label encoding except the fact that in ordinal encoding the categorical feature is encoded according to some sort of hierarchy in the system, For example if there are three categories in the categorical feature named - "High-School", "Bachelor's", and "Master's" the ordinal encoding will label this as 0, 1, 2 based on the educational hierarchy.

The choice of the encoding method depends on the nature of the categorical feature, the machine learning algorithm that we are using and the specific requirements of the given project.

---

### Handling Skewed Distribution
Many machine learning algorithms assumes that the data features are normally distributed, this is why handling skewed distribution becomes an essential task in data transformation process, as the skewed data might lead to biased or inaccurate model. As we have seen transformers in the Dealing with Outliers process of this article they are used usually to normally distribute the features, let's discuss some of the most common transformation techniques which will be used to handle skewed data:

**Logarithmic Transformation:**  
This transformation technique is used when the data feature is right skewed or positively skewed. This transformation applies natural log values to all the data points, and thus it must be noticed that this technique only works on positive values only, as the log of positive values could only be taken.

**Square Root Transformation:**  
Square root transformation is usually used in moderately right skewed data, it is less effective as compared to logarithmic transformation and it takes square root of each data point and makes it more like the data is normally distributed.

**Box-Cox Transformation**  
Box-Cox Transformation is more suitable for right skewed data with data points either being positive or zero valued. It uses a parameter \( \lambda \) which helps in finding the best approximation to approximate the normality.

**Yeo-Johnson Transformation**  
Yeo-Johnson transformation is more effective in nature, it works in a similar way as Box-Cox transformation, but it can work for both right as well as left skewed data feature, it can also work for either positive or negative value, which the box-cox transformation lacks.

**Qunatile Transformation**  
This type of transformation works both for right skewed as well as left skewed data variable. Here the data is distributed according to there percentile in which they lay, hence, distributing the data into uniform sets.

The choice of transformation technique depends on the data we are working on and the skewness of the data, it is always preferred to visualize the data before applying the transformation technique.

---

### Feature Engineering
The process of creating new features or modifying the existing feature to improve the performance of machine learning model is called feature engineering. It helps in creating more informative and effective representation of patterns present in data by combining and transforming the given features. Through feature engineering we can increase our model performance and generalization ability. We have already seen some of the feature engineering techniques such as binning and normalization in previous steps, let's discuss some of the other most important techniques which we haven't discusses:

**Polynomial Features**  
We can capture non linear relationships in the data by taking polynomial features into consideration, by squaring, cubing or increasing the power of the feature present. This technique can add flexibility to basically the linear models that are well known like linear regression, logistic regression etc. We can apply regularization with polynomial features to reduce the risk of overfitting.

**Interaction Terms**  
We can combine two or more features together to produce a new feature, this helps machine learning algorithms specially linear models to identify and leverage the combined effect of different features on the outcome. The interaction terms uncovers the patterns that are not focused if individual features are considered, these terms helps in understanding the relationship between different variables and the effect of change in one feature on the behaviour of another. For example suppose we are modelling a simple regression problem of house price prediction, there are different house whose length and width of span is given let them be 'l', 'b' respectively. It is better to introduce a new feature area which is the multiplication on length and width, or 'l.b', which is a better indication of house price.

**Domain-Specific Feature**  
We must consider creating features that are highly relevant and informative about the problem in hand. This process involves deep understanding of the domain on work we are in, as well as the knowledge of the data presented to us. This helps us create new features that might not be that much of use immediately but is essential for domain we are currently performing analysis in.

---

### Dimensionality Reduction
Dimentionality reduction is the process of reducing the number of features in the dataset while preserving the information that the original dataset conveys. It is often considered good to reduce the dimensions of highly dimensional dataset to reduce the computational complexity and reduce the chances of overfitting. There are two dimensionality reduction techniques which are used widely.

**Principal Component Analysis(PCA)**  
PCA is the most common dimensionality reduction technique used in machine learning which transforms higher dimension data into lower dimension data retaining the information of the original dataset. PCA deals with the generation of principal components through standardization of data, finding covariance matrix of the data and then arranging the eigenvector obtained from the covariance matrix according to eigen values in descending order. In PCA the original data is projected onto the principal components to obtain lower dimensional data.

**t-Distributed Stochastic Neighbor Embedding**  
t-SNE reduces the dimensionality of the data while maintaining local relationships between data points. What t-SNE algorithm does is that it takes higher dimensional data and finds out the similarities in between the data points, such that if this data point occurs what is the probability of the other data point occurring, and then it does the same with lower dimensional data and tries to reduce the divergence between the pairwise data points in high and low dimension space.

---

### Text Data Transformation
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

### Advantages and Disadvantages of Data Transformation
There are several advantages of using data transformation, but with positive points there are also negative points that we must pay attention to such that we can achieve our goals that we have set from each project in hand. Let's discuss some of the advantages as well as disadvantages of data transformation that we must pay attention to such that we can best use our knowledge to improve our model's performance:

**Advantages**
- Improved Model Performance: The model gets better at generalizing new data when the issues in data are resolved through data transformation.
- Handling Missing Data: Results into good increase in the accuracy of the model.
- Better Convergence: Data normalization and standardization results into better convergence of the model during it's training period.
- Dimensionality Reduction: Simplifies the model training process.
- Better Insights from Feature Engineering

**Disadvantages:**
- Information Loss: If the transformation is not within limit valuable details might get discarded leading to a failed model.
- Risk of Overfitting: Excessive feature engineering and complex transformations might lead to overfitting, the model might fit data too closely and perform bad on new, unseen cases.
- Data Leakage: Applying transformation inappropriately using the entire dataset including the test data can lead to overestimation of model performance.
- Increased Complexity
- Assumption Violation: Sometimes transformation might not align with the assumptions of the chosen machine learning algorithm, which might lead to bad performance of the model in general.

Thus we can say that data transformation is a crucial step in machine learning, it requires careful consideration before deciding which techniques to apply to the data. While data transformation can improve model performance, it's important to avoid potential pitfalls such as information loss and overfitting. Cross-validation and evaluating model performance on unseen data are essential steps to ensure that the chosen data transformation techniques are appropriate and effective for the given task.

---

## Conclusion
In conclusion, data transformation is vital for refining raw data, improving model performance, and ensuring accurate machine learning outcomes.
```markdown

---

