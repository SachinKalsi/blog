---
tags: 'machine learning, mathematics, key performance indicators (KPIs)'
published: true
min_to_read: 10
---

There are various machine learning algorithms like KNN, Naive Bayes, Logistic Regression, Linear Regression, SVM, Ensemble models etc etc. But how do you measure the performance of these models ? Is there any one metric which will define how good a model is ? Well, in this blog I will discuss about the important KPIs (Key Performance Indicators) to measure the performance of a Machine Learning Model


1. Accuracy
2. Confusion matrix
3. Precision, Recall & F1 score
4. Area Under The Curve (AUC)
5. Log-loss
6. R-Squared Coefficient
7. Median absolute deviation (MAD)
8. Error Distribution
  
__Sample dataset__: I will consider a sample test dataset to explain the metrices. The sample data contains 15 positive Y's & 15 negative Y's. It is a simple binary classification problem 

<p class='note'> P.S: We measure the performance of models based only on TEST data and NOT on train data/ cross-validation data. </p>

## 1. Accuracy

Accuracy is the most easiest metric to understood & to implement as well. Accuracy is nothing but the ratio of correctly classified data points to the total number of data points.

__Example__: For the above sample dataset, if a model predicts 13 positive points & 14 negative points correctly, then

$$Accuracy = (13+14)/30$$

$$Accuracy = 0.9$$

<p class='note'>Cons</p>

1. __Accuracy is a bad metric for a highly imbalanced dataset__: Consider a highly imbalanced dataset like credit card fraud detection, where >99% of the data points are negative. Lets say we have very dumb model which will always predict negative, irrespective of given data points (i.e., X). And if we calculate the Accuracy for this dumb model, it would be more than 0.99 that means even a dumb model is showing an accuracy of 0.99 which is obviously wrong.

2. __Can not take probability scores__: Accuracy don't take probability scores into consideration. It will just consider the predicted Y values. For __example__

Lets say we have two Models M1 & M2.

|  ID  |  Y  | M1_prob  |  M2_prob  |  M1_pred  | M2_pred |
| :---:| :---:|:---:|:---: |:---: |:---: |
|   1  |  1  |  0.90 | 0.55 |  1  |  1   |
|   2  |  0  |  0.10 | 0.45 |  0  |  0   |
|   3  |  1  |  0.88 | 0.51 |  1  |  1   |
|   4  |  1  |  0.93 | 0.60 |  1  |  1   |
|   5  |  0  |  0.01 | 0.48 |  0  |  0   |

Where,

    Y : real Y values
    M1_prob: probability of Y = 1 for model M1
    M2_prob: probability of Y = 1 for model M2
    M1_pred: predicted Y values from model M1
    M2_pred: predicted Y values from model M2
    
As we can see, both M1 & M2 models predicted the same Y's & have same accuracy.But if you see probability scores of M1 & M2, the model M1 is performaning better than the model M2, since the probability scores are high for the M1 model, whereas for the model M2,the probability scores are almost half


## 2. Confusion matrix:

__Coming soon__