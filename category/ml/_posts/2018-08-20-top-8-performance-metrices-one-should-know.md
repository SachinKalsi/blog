---
tags: 'machine learning, mathematics, key performance indicators (KPIs)'
published: true
image: 'data/images/blog_posts/kpi.jpg'
min_to_read: 10
---

There are various machine learning algorithms like KNN, Naive Bayes, Logistic Regression, Linear Regression, SVM, Ensemble models etc etc. But how do you measure the performance of these models ? Is there any one metric which will define how good a model is ? Well, in this blog I will discuss about the important KPIs (Key Performance Indicators) to measure the performance of a Machine Learning Model


1. <A href="#accuracy">Accuracy</A>
2. <A href="#confusion-matrix">Confusion matrix</A>
3. <A href="#precision-recall-f1-score">Precision, Recall & F1 score</A>
4. <A href="#area-under-the-curve-auc">Area Under The Curve (AUC)</A>
5. Log-loss
6. R-Squared Coefficient
7. Median absolute deviation (MAD)
8. Error Distribution
  
__Sample dataset__: I will consider a sample test dataset to explain the metrices. The sample data contains 15 positive Y's & 15 negative Y's. It is a simple binary classification problem 

<p class='note'> P.S: We measure the performance of models based only on TEST data and NOT on train data/ cross-validation data. </p>


## Accuracy

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


## Confusion matrix: 

Confusion matrix is a matrix used to describe the performance of a classification model. It solves the problem the accuracy metric had with the imbalanced dataset

Lets consider a binary classification problem i.e., Y belongs to 0 or 1. 0 being negative label & 1 being positive label. Let Y be the actual values & Y^ be the predicted values

![confusion_matrix]({{site.baseurl}}data/images/confusion_matrix.png)

where,

__TN (True Negative)__  :  number of data points for which Y = 0 & Y^ = 0

__FN (False Negative)__ :  number of data points for which Y = 1 & Y^ = 0

__FP (False Positive)__ :  number of data points for which Y = 0 & Y^ = 1

__TP (True Positive)__  :  number of data points for which Y = 1 & Y^ = 1

<i class='note'>Tip to remember: </i> Here is a tip to remember which I learned from <a target="_blank" href="https://www.appliedaicourse.com/">Applied AI course</a>

![confusion_matrix_tip]({{site.baseurl}}data/images/cm_tip.png)

__Example__: Lets consider the case Y = 0, Y^ = 0. Here predicted value is 0, i.e, Negative (N)
 & the it is correct (T) (since the actual value is also Negative). So it becomes __True Negative (TN)__
 
If we have a multi class classification, then confusion matrix size would be CxC, where C is the number of class labels.

![confusion_matrix_multi_class_classification]({{site.baseurl}}data/images/cm_multi.png)

If the model you trained is very good, then all the <a href="https://en.wikipedia.org/wiki/Main_diagonal" target="_blank">principle diagonal values</a> will be high & non-principle diagonal values will be low or zero.

Let P & N be the total number of Positive & Negative test data points respectively, Then

$$TPR (True Positive Rate) = TP/P$$

$$TNR (True Negative Rate) = TN/N$$

$$FPR (False Positive Rate) = FP/P$$

$$FNR (False Negative Rate) = FN/N$$

In layman language, 

TPR = Percentage of correctly classified positive points
 
TNR = Percentage of correctly classified negative points

FPR = Percentage of incorrectly classified positive points

FNR = Percentage of incorrectly classified negative points.

__Example__: Consider a highly imbalanced dataset like credit card fraud detection, where >99% of the data points are negative. Lets say we have very dumb model which will always predict negative, irrespective of given data points (i.e., X). And if we calculate the Confusion matrix for this dumb model, then TNR will be 100 & TPR will be zero but FNR will be high which indicats the model is very dumb

In Confusion matrix we have 4 numbers, how one will know which one is important? Well, it depends on the dataset.

__Example__: If the dataset is cancer detection/credit card fraud detection, then we want high TPR and very less or 0 FNR. 

<p class='note'>Cons</p>

1. __Can not take probability scores__: Confusion matrix don't take probability scores into consideration. It will just consider the predicted Y values. 

## Precision, Recall, F1 score

Precision & Recall are used in information retrieval, pattern recognition & binary classification. Please refer <a target="_blank" href="https://en.wikipedia.org/wiki/Precision_and_recall">Wiki</a> for info. Precision & Recall considers only the positive class labels.

$$Precision = \frac{TP}{FP+TP}$$

$$Recall = \frac{TP}{FN+TP}$$

In layman language,

__Precision__: Of all the predicted POSITIVE labels, how many of them are actually POSITIVE

__Recall__: Of all the actual POSITIVE labels, how many of them are correctly predicted as POSITIVE.

We always want Precision & Recall to be high. Precision & Recall values ranges from 0 to 1

__F1 Score__: It is a single measurement, which considers both Precision & Recall to compute the score. It is basically the harmonic mean of Precision & Recall. The value of F1 score ranges from 0 to 1.

$$F1 score = 2 *\frac{Precision*Recall}{Precision+Recall}$$

This metric is extensively used in <a target="_blank" href="https://www.kaggle.com/">Kaggle</a> compitations. But Precision & Recall is more interpretable or I can say F1 score is difficult to interpret.

## Area Under The Curve (AUC)

Coming soon