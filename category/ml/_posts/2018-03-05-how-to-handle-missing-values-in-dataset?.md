---
tags: 'machine learning, missing values in features'
published: true
---

Missing values in a dataset is very much common in real time. Taking action on such missing values is inevitable because missing data causes problems.

## Lets take a sample dataset.

|  ID  |  F1  | F2  |  F3  |  F4  | class label |
| :---: |:---: |:---:|:---: |:---: |:---: |
|   1  |  12  |  6  |  1  |  89   |  0   |
|   2  |  9   |  8  |  0   |  76  |  1   |
|   3  |  14  |  11 | None |  78  |  1   |
|   4  |  10  | None|  2   |  90  |  0   |
|   5  |  11  |  9  |  3   |  66  |  1   |
|   6  |  8   |  13 |  5   |  90  |  0   |
|   7  |  12  |  10 |  1   |  72  |  1   |

___
- The sample data set contains 7 data points & 4 features and 1 class label.
- A cell is represented as (row_id, column_id). Features F1 ,F2, F3, F4 corresponds to column id of 1,2,3,4 respectively
- As we can see the values for the cells (3,3) & (4,2) are not given. (None means empty value)

Here I will be discussing some techniques to handle the missing values.

## 1. Remove rows (data points) with missing values

This is the simplest strategy to handle missed values

## 2. Imputation (Replacing the missed values)

- Replacing the missed values by **mean (average) or median or mode** (most frequently occurred value)

    __Example__:

    **_If we choose to replace the missed value by mean, then the value of the cell (3,3) becomes 2. (Average of 1,0,2,3,5,1 is 2)_**

- Impute based on class labels.

    Instead of considering all the data points for the calculation of mean or median or mode, consider only data points whose class label matches with the class label of missed value data point.

    __Example__:

    (3,3) is a missed value. The class label of 3rd row is 1. The row id's which are having class labels as 1 are 2, 5, 7

    **_If we choose to replace the missed value by mean, then the value of the cell (3,3) becomes 1. (Average of 1,3,1 is 1)_**

## 3. Source of information

Missed values can also become source of information.

__Example__: Lets say we are collecting **age, sex, weight, height, hair color** of all peoples in a village. Here hair color is an interesting attribute. Some people might have grey color hair, black color hair, white color hair etc. But what about the people who do not have hairs at all? These people will leave the field **hair color** empty.

So if hair color field is empty, then it says that particular person is not having hairs & it could be an important information to note.

So we store these information in separate fields as shown below.

**Lets take a subset of the sample data**

|  ID  |  F1  | F2  |  F3  |  F4  | class label |
| :---: |:---: |:---:|:---: |:---: |:---: |
|   2  |  9   |  8  |  0   |  76  |  1   |
|   3  |  14  |  11 | None |  78  |  1   |
|   4  |  10  | None|  2   |  90  |  0   |

**Imputed data**

|  ID  |  F1  | F2  |  F3  |  F4  | class label |
| :---: |:---: |:---:|:---: |:---: |:---: |
|   2  |  9   |  8  |  0   |  76  |  1   |
|   3  |  14  |  11 | **<span style='color:red'>1</span>** |  78  |  1   |
|   4  |  10  | **<span style='color:red'>10</span>**|  2   |  90  |  0   |

We use **Missing value features (<span style='color:red'>Binary</span>)** to indicate the missing of data values. We use **1** if the value is missing else **0**

|  ID  |  F1  | F2  |  F3  |  F4  | class label |
| :---: |:---: |:---:|:---: |:---: |:---: |
|   2  |  0   |  0  |  0   |  0  |  0   |
|   3  |  0  |  0 | **<span style='color:red'>1</span>** |  0  |  0   |
|   4  |  0  | **<span style='color:red'>1</span>**|  0  |  0  |  0   |

## 4. Model based Imputation

In this technique, we assume the feature, for which the values are missing for some data points, as class label & we predict the missed value using the algorithms like K-NN etc.

__Example__: Lets assume that, we got the value for the cell (3, 3) as 1 (by using Impute based on class labels technique). Now sample data looks like below.


|  ID  |  F1  | F2  |  F3  |  F4  | class label |
| :---: |:---: |:---:|:---: |:---: |:---: |
|   1  |  12  |  6  |  1  |  89   |  0   |
|   2  |  9   |  8  |  0  |  76  |  1   |
|   3  |  14  |  11 |  1 (_Imputed_)  |  78  |  1   |
|   4  |  10  | <span style="color:red">None</span>|  2  |  90  |  0   |
|   5  |  11  |  9  |  3  |  66  |  1   |
|   6  |  8   |  13 |  5  |  90  |  0   |
|   7  |  12  |  10 |  1  |  72  |  1   |

As we can see, **_F2_** is the feature for which the value is missing at the cell (4,2). So **_F2_** becomes the class label. Now we need to predict the value for the cell (4,2). Modified data set looks as shown below.

|  ID  |  F1  | <span style="color:red">F (earlier **_class label_**)</span> |  F3  |  F4  | <span style="color:red">class label (earlier **_F2_**)</span> |
| :---: |:---: |:---:|:---: |:---: |:---: |
|   1  |  12  |  0  |  1  |  89   |  6   |
|   2  |  9   |  1  |  0  |  76  |  8   |
|   3  |  14  |  1 |  1  |  78  |  11   |
|   5  |  11  |  1  |  3  |  66  |  9   |
|   6  |  8   |  0 |  5  |  90  |  13   |
|   7  |  12  |  1 |  1  |  72  |  10   |
|   4  |  10  | 0|  2  |  90  |  <span style="color:red">None</span>   |

Now using an algorithm like KNN, we can predict the missed values.

___

Please <a href="mailto:sachinkalsi15@gmail.com">contact me</a> regarding any queries or suggestions
