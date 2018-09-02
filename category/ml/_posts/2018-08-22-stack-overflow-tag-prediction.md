---
tags: 'machine learning, stackoverflow, casestudy'
published: true
image: 'data/images/blog_posts/stackoverflow.png'
min_to_read: 20
---

If you are a software engineer or a programmer you must have used [stackoverflow](https://stackoverflow.com){:target="_blank"} atleast once in your life time. But have you ever wondered how stackoverflow predicts the tags for a given question ? In this blog, I will discuss about <b>stackoverflow tag predictor</b> case study which I have done as a part of the course in [appliedaicourse](https://www.appliedaicourse.com/){:target="_blank"}

---

## Problem Statement

Predict the tags (a.k.a. keywords, topics, summaries), given only the question text and its title.

Read the full problem statement on [kaggle](https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/){:target="_blank"} 

## Business Objectives and Constraints

1. Predict tags with high [precision and recall](https://goo.gl/csnXGo){:target="_blank"}
3. No strict latency constraints.

---

## Machine Learning Problem

### Data

Data contains 4 fields

1. Id - Unique identifier for each question

2. Title - The question's title

3. Body - The body of the question

4. Tags - The tags associated with the question

Click [here](https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/data){:target="_blank"} for more details.

<b>Size of Train.csv</b> - 6.75GB

<b>Number of rows in Train.csv</b> = 6034195

<i><b>P.S:</b>There are two files: Train.csv & Test.csv. Based on the computing power of my system, I have taken only a subset of data from Train.csv </i>


### Sample data point

<b>Id</b>: 5

<b>Title:</b> How to modify whois contact details?

<b>Body:</b>

<pre>
&lt;pre&gt;
    &lt;code&gt;
    function modify(.......)
    {
        $mcontact = file_get_contents( "https://test.httpapi.com/api/contacts/modify.json?auth-userid=$uid&amp;auth-password=$pass&amp;contact-id=$cid&amp;name=$name &amp;company=$company&amp;email=$email&amp;address-line-1=$street&amp;city=$city&amp;country=$country&amp;zipcode=$pincode&amp;phone-cc=$countryCodeList[$phc]&amp;phone=$phone" );
        $mdetails = json_decode( $mcontact, true );
        return $mdetails;
    }
    &lt;/code&gt;
&lt;/pre&gt;
&lt;p&gt;using this modify function, displays warning mesage&lt;/p&gt;
&lt;pre class="lang-none prettyprint-override">
    &lt;code&gt;
    Warning: file_get_contents(https://...@hihfg.com&amp;address-line-1=3,dfgdf,fgdf&amp;city=dfgfd&amp;country=India&amp;zipcode=641005&amp;phone-cc=91&amp;phone=756657)
        [function.file-get-contents]: failed to open stream: HTTP request failed!
        HTTP/1.0 400 Bad request in /home/gfdgfd/public_html/new_one/customer/account/class.whois.php on line 49\n
    &lt;/code&gt;
&lt;/pre&gt;
&lt;p&gt;Please help me, modify contact details.&lt;/p&gt;
</pre>

<b>Tags</b>: php api file-get-contents

---
## Mapping the Business problem to a Machine Learning Problem 

### Type of Machine Learning Problem

<b><i>It is a multi-label classification problem.</i></b>

In Multi-label Classification, multiple labels (in this problem its tags) may be assigned to each instance and there is no constraint on how many of the classes the instance can be assigned to.
Source: [Wiki](https://en.wikipedia.org/wiki/Multi-label_classification){:target="blank"}

Find more about multi-label classification problem [here](http://scikit-learn.org/stable/modules/multiclass.html){:target="blank"}

A question on Stackoverflow might be about any of C, Pointers, JAVA, Regex, FileIO and/or memory-management at the same time or none of these.

### Performance metric

1. Micro F1 score
2. Macro F1 score:

I have already discussed in detail about the performance metrices in my previous [blog](https://goo.gl/Z4AP95){:target="blank"} post. Please read [this](https://goo.gl/UJdNps){:target="blank"} blog to know more about Micro & Macro F1 scores.

---
## EDA (Exploratory Data Analysis)

I have used [pandas](https://pandas.pydata.org/){:target="blank"} library to load the data. Please visit my github repo to see the full code. I have taken a sample of 1000000 (10 lakh) data points from Train.csv. Here is a list of major observations from EDA.

1. <b>Number of rows in the database:</b> 6034196
2. <b>5.6% of the questions are duplicate:</b> Number of rows after removing duplicates:  943582
3. <b>Number of unique tags:</b> 34945
4. <b>Top 10 important tags:</b>  ['.a', '.app', '.aspxauth', '.bash-profile', '.class-file', '.cs-file', '.doc', '.drv', '.ds-store', '.each']

5. Few number of tags have appeared more than 50000 times & the top 25 tags have appeared more than 10000 times
![Distribution of number of times tag appeared in questions(for first 100 tags)]({{site.baseurl}}data/images/stackoverflow/tag_counts.png)
6. <b>Tags analysis</b>
    1. Maximum number of tags per question: 5
    2. Minimum number of tags per question: 1
    3. Avg. number of tags per question: 2.887779
    4. Questions with 3 tags appears more in number
    ![Number of tags in the question]({{site.baseurl}}data/images/stackoverflow/question_with_tag_frequency.png)
    5. Word cloud of tags
    ![Word cloud of tags]({{site.baseurl}}data/images/stackoverflow/word_cloud.png)
    6.`C#` appears most number of times, `Java` is the second most. Majority of the most frequent tags are programming language. And here is the chart for top 20 tags
    ![frequency of top 20 tags]({{site.baseurl}}data/images/stackoverflow/frequency_of_top_20_tags.png)

---    
    
## Cleaning and preprocessing of Questions

<i>P.S: <b>Due to hardware limitations, I am considering only 500K data points</b></i>

### preprocessing
<ol>
  <li>56.37% percentage of questions contains HTML tag &lt;code&gt; tag. So separate out code-snippets from  the Body</li>
  <li>Remove Spcial characters from title and Body (not in code)</li>
  <li><b>Remove stop words (Except 'C')</b></li>
  <li>Remove HTML Tags</li>
  <li>Convert all the characters into small letters</li>
  <li>Use SnowballStemmer to stem the words.<br><br>
  <i>Stemming is the process of reducing a word to its word stem. <br>
  <b>For Example:</b> "python" is the stem word for the words ["python" "pythoner", "pythoning","pythoned"]</i></li>
  <li><b>Give more weightage to title: Add title three times to the question</b>. Title contains the information which is more specific to the question and also only after seeing the question title, a user decides whether to look into the question in detail. At least most of the users do this if not all </li>
</ol>

<h5>Sample question after preprocessing:</h5>

<i>"modifi whoi contact detail modifi whoi contact detail modifi whoi contact detail use modifi function display warn mesag pleas help modifi contact detail"</i>

---
## Machine Learning Models

<i>
Total number of questions: 500000<br>
Total number of tags: 30645
</i>

Here we are going to use <i><b>Problem Transformation(Binary Relevance)</b></i> method to solve the problem.

<h5>Binary Relevance:</h5> Here we are going to convert multi-label classification problem to multiple single class classification problems. Basically in this method, we treat each label (in our case its tag) as a separate single class classification problem. This technique is simple and is widely used.

Please refer to [analytics vidhya's blog](https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/){:target="_blank"} to know more about the techniques to solve a Multi-Label classification problem

---

<i>Coming soon</i>
 