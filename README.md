# PROJECT OVERVIEW: Kaggle's Fake News Detection Challenge (Classification Modelling with Python)

## Table of Contents
* [Introduction](#introduction)
* [Code and Resources Used](#code-and-resources-used)
* [Binary Classification Introduction](#binary-classification-introduction)
* [Kaggle Dataset](#kaggle-dataset)
* [Data Pre-Processing](#data-pre-processing)
  * [Part 1: Column Operations](#part-1-column-operations)
  * [Part 2: Natural Language Processing](#part-2-natural-language-processing)
* [Training Different Models and Creating a Predictive Model](#training-different-models-and-creating-a-predictive-model)
  * [Part 1: Train-Test Split](#part-1-train-test-split)
  * [Part 2: Training Different Classification Models](#part-2-training-different-classification-models)
  * [Part 3: Developing a Predictive Model](#part-3-developing-a-predictive-model)
* [Discussion](#discussion)

---

# Introduction
  
  <ul>
    <li>Undertook Kaggle's Fake New Challenge, and created a binary classification model to detect fake news.</li>
    <li>Tested out the following classification models: Logistic Regression, SVM, Random Forest Classifier, K-Nearest Neighbour Classifier, Decision Tree Classifier.</li>
    <li>Link to Kaggle's Fake News Challenge, and downloadable data sets: https://www.kaggle.com/competitions/fake-news/overview </li>
  </ul>

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Code and Resources Used

  <ul>
    <li><b>IDEs Used:</b> Google Colab, Jupyter Notebook</li>
    <li><b>Python Version:</b> 3.10.12</li>
    <li><b>Libraries and Packages:</b>
    <ul>
      <li><b>Libraries for data manipulation: </b> pandas, numpy </li>
      <li><b>Libraries for Natural Language Processing: </b> 
        <ul>
          <li>re</li>
          <li>nltk</li>
          <li>stopwords (from nltk.corpus)</li>
          <li>PorterStemmer (from nltk.stem.porter)</li>
        </ul></li>
      <li><b>Libraries for binary classification modelling: </b> 
      <ul>
          <li>TfidfVectorizer (from sklearn.feature_extraction.text)</li>
          <li>train_test_split (from sklearn.model_selection)</li>
          <li>LogisticRegression (from sklearn.linear_model) </li>
          <li>svm (from sklearn)</li>
          <li>SVC (from sklearn.svm) </li>
          <li>RandomForestClassifier (from sklearn.ensemble)</li>
          <li>KNeighborsClassifier (from sklearn.neighbors)</li>
          <li>DecisionTreeClassifier (from sklearn.tree)</li>
          <li>accuracy_score (from sklearn.metrics)</li>
      </ul></li>
    </ul></li>
  </ul>

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Binary Classification Introduction

<p>Classification is a type of supervised machine learning algorithm that attempts to correctly assign a label given an input. For example, in image classification, the machine learning model attempts to 
label an image, such as whether the image input is that of a dog or a cat. What was just described is a classic example of binary classification, where the machine predicts whether the input belongs in one category or the other (i.e., one of possible two). Determining whether an article can be classified as fake news or not, as this project aims to do, is another example of binary classification.</p>

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Kaggle Dataset

The Kaggle dataset contains:
<ul>
<li>train.csv: A full training dataset. This dataset will be used.</li>
<li>test.csv: A full testing dataset, without the labels.</li>
<li>submit.csv: A sample submission.</li>
</ul>

The training dataset has the following attributes:
<ul>
<li><b>id:</b> unique id for a news article</li>
<li><b>title:</b> the title of a news article</li>
<li><b>author:</b> author of the news article</li>
<li><b>text:</b> the text of the article; could be incomplete</li>
<li><b>label:</b> a label that marks the article as potentially unreliable (1=unreliable, 0=reliable)</li>
</ul>

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Data Pre-Processing

We will be using the training dataset.

#### Part 1: Column Operations

![image](https://github.com/user-attachments/assets/f2172cf4-9116-46e7-95f0-00b6cce8d598)

<ul>
<li><b>STEP 1:</b> Check for null values for each column, then replace null values with a blank. (news_df.fillna(''))</li>
<li><b>STEP 2:</b> Append "author" and "title" column into one column called "content" (news_df['content']=news_df['author']+' '+news_df['title'])</li>
</ul>

#### Part 2: Natural Language Processing

Machine learning algorithms are unable to completely understand the complexities of textual nuances such as special characters (e.g., ', ! ), capital letters, words derived from a root word. As machine learning algorithms understand inputs numerically, it is therfore imperative that operations are performed to ensure that we minimize the disporportionate effect that certain words or letters would have on numerical value assignment as we build this predictive model.

<ul>
<li><b>STEP 1:</b> Set PortStemmer() equal to a variable (in this case, port_stem). This library will extract root words (e.g. it will extract the root word "read" from words such as "unread" and "read".</li>
<li><b>STEP 2:</b> Create the following function, which intends to remove the effects that certain nuances of written text will have on numerical value assignment, as mentioned in the preamble of this part.</li>
  
  ![image](https://github.com/user-attachments/assets/82c88d03-e9ce-4583-943d-fb228ba45f25)

<li><b>STEP 3:</b> Use the apply function on the new_df's "content" column.</li>
</ul>

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Training Different Models and Creating a Predictive Model

#### Part 1: Train-Test Split
Code snippet for setting X and Y for the train-test split:

![image](https://github.com/user-attachments/assets/a20ae234-da1e-42f2-9050-38a8ded34b64)

Recall that machine learning models run more smoothly with numerical inputs. Use the following code snippet to vectorize X into numerical data:

![image](https://github.com/user-attachments/assets/c7035346-ce5d-4bb9-997f-77088008b940)

Now we can set up the train-test split, with test size being 20%, stratification on Y to preserve proportionality of 1's and 0's in both the training and testing datasets, and random state set to any integer such that reporducibility is preserved (i.e., keep the randomness of selection under control).

![image](https://github.com/user-attachments/assets/abc1d70d-b25b-47a3-83a4-46d15c49ee26)


#### Part 2: Training Different Classification Models
There are many different classification models that can be applied to binary classification problems, but I will perform training on some of the more commen classification models employed in binary classification. This may not necessarily mean that these are the most well-performing models for this problem. In fact, one of these models had quite a low accuracy score.

Sample code snippet:

![image](https://github.com/user-attachments/assets/040d5607-eb6e-4724-bdb4-4ae594d9738f)

![image](https://github.com/user-attachments/assets/e447b7c1-a1d7-493c-8004-479eac5d40fb)

![image](https://github.com/user-attachments/assets/f64d00f2-595e-4cab-a3d9-b309b73e0fe0)

<ul>
<li><b>Logistic regression:</b> </li>
  <ul>
    <li>Library: LogisticRegression() </li>
    <li>Training data accuracy score: 0.9863581730769231</li>
    <li>Testing data accuracy score: 0.9790865384615385</li>
  </ul>
<li><b>Support Vector Machine:</b> </li>
  <ul>
    <li>Library: SVC() </li>
    <li>Training data accuracy score: 0.9990985576923077</li>
    <li>Testing data accuracy score: 0.9889423076923077</li>
  </ul>
<li><b>Random Forest Classifier:</b> </li>
  <ul>
    <li>Library: randomForestClassifier()</li>
    <li>Training data accuracy score: 1.0</li>
    <li>Testing data accuracy score: 0.9942307692307693</li>
  </ul>
<li><b>K-Nearest Neighbour Classifier:</b> </li>
  <ul>
    <li>Library: knn() </li>
    <li>Training data accuracy score: 0.5360576923076923 </li>
    <li>Testing data accuracy score: 0.5233173076923077</li>
  </ul>
<li><b>Decision Tree Classifier:</b> </li>
  <ul>
    <li>Library: dt() </li>
    <li>Training data accuracy score: 1.0</li>
    <li>Testing data accuracy score: 0.9913461538461539</li>
  </ul>
</ul>

Other than KNN, each model performed with a high level of accuracy. However, many of them took longer to execute. Even though most models had higher accuracy scores than the logistic regression model, the difference was marginal at best. It would not be efficient to use models that have a slightly higher accuracy score but a much longer computational time.

Therefore, logistic regression has been determined to be the best classification model moving forward.

#### Part 3: Developing a Predictive Model

Code snippet:

![image](https://github.com/user-attachments/assets/7fb16cb9-75bf-471f-8fe9-75aed1a4229b)

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Discussion

<p>The model that has been build appears to be very accurate in detecting whether a news article can be classified as fake news or not. It will need refinements if it were to be reproducible and applied for news coverage beyond the Kaggle dataset. For instance, as the 2025 midterm elections and federal general elections in the Philippines and Canada, respectively, are approaching, it is of utmost importance to develop mechanisms to quickly detect fake news. It has been argued that the proliferation of fakes news was a major factor in securing President Ferdinand Marcos Jr.'s controversial victory in the 2022 Philippine Presidential Election. In the age of advancements in Deepfake technologies and Generative AI, the ability to detect fake news must outpace the proliferation of it.</p>

[<b>Back to Table of Contents</b>](#table-of-contents)


