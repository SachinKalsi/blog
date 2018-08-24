---
tags: 'machine learning, mathematics, tf idf, bow, vectors'
published: true
min_to_read: 4
---
In order to apply any Machine Learning algorithm, we have to convert all non-numerical data to numerical data.

Here I am going to discuss some of the methodolgies to convert text data into numerical vectors. A vector is nothing but a numerical array. Please look into [Linear Algebra]({{'/blog/category/ml/2018/02/25/linear-algebra-for-machine-learning.html' }}) section to know more about vectors & their properties.

Also by converting text data to numerical array, we can measure similarity between the texts by using <a target="_blank" href="https://sachinkalsi.github.io/blog/category/ml/2018/02/25/linear-algebra-for-machine-learning.html#euclidean-distance" > Euclidean distance </a> or Cosine similarity

Lets consider the following example.

1. text1: <b>India is a beautiful country</b>

2. text2: <b>I am proud of my country</b>

## <b>BOW model (Bag Of Words)</b>
  
In this model, each text data is converted to a vector whose length is equal to number of unique words in the whole document. Initially each value in the vector is initilised to zero. 

```
For both text1 & text2, the vector looks as follows initially
```

| I | India | a | am | beautiful | country | is | my | of | proud
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|0|0|0|0|0|0|0|0|0|0

#### Calculate BOW values

```
For each of the word in the text_data:
  increase the count of the word by 1
```

  (i) text1: (India is a beautiful country)
  
  | I | India | a | am | beautiful | country | is | my | of | proud
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  |0|1|1|0|1|1|1|0|0|0
  
  
  (ii) text2: (I am proud of my country)
  
  | I | India | a | am | beautiful | country | is | my | of | proud
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  |1|0|0|1|0|1|0|1|1|1
  
  If we have very large number of unique words, then the size of the vector will be very large
  
## <b>TF - IDF (Term frequency - Inverse Document Frequency)</b>

#### TF (Term Frequency)
  
  TF is measured for a word in the text data. TF for a word in text data is the probability of that word in the text data.i.e., 
  
![]({{site.baseurl}}data/images/tf.png)

#### IDF (Inverse Document Frequency)

  IDF is measured for a word with respect to the whole document. IDF of a word $$W_j$$ is defined as the log of total number of documents in the corpus to the number of documents that contains   the word $$W_j$$. i.e.,
  
![]({{site.baseurl}}data/images/idf.png)

#### TF-IDF

It is the product of TF & IDF of a word.

So TF-IDF value balances between the rare words in the document & high frequency words in the corpus. High value of IDF indicates, the word occurs very few times in the whole document corpus. And a high value of TF indicates, the word occurs more number of times in the text document.

  1. TF-IDF Vector representation of text1 (India is a beautiful country)
  
  $$TF-IDF(India) = TF(India, text1) * IDF(India, corpus)$$
  
  $$= (1/5) * log(2/1)$$
  
  $$= 0.06$$
  
  Similary we can calculate TF-IDF values for other words. So TF-IDF vector for text1 can be represented as follows
  
  | I | India | a | am | beautiful | country | is | my | of | proud
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  |0|0.06|0.06|0|0.06|0|0.06|0|0|0
  
  
  There are also many variations of TF-IDF model like TF-IDF word2vec model. I will discuss word2vec in Deep Learning section (...coming soon...)
  