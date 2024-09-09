---
title: "Sentiment Analysis : Naive Bayes Classifier"
date: 2022-07-12
categories: [Natural_Language_Processing, Text_Classification]
tags: [machine-learning, text-classification, sentiment-analysis, naive-bayes,nlp]
comments: false
math: true
---

Classification is a fundamental aspect of both human and machine intelligence. In this blog, we'll explore a specific text classification problem: sentiment analysis. The most common approach to text classification in natural language processing involves supervised machine learning. Formally, the task of supervised classification is to take an input $x$ and a fixed set of output classes $Y = \{y_1, y_2, \dots, y_m\}$, and return a predicted class $y \in Y$. In the context of text classification, we often refer to the output variable as c (for "class") and the input variable as d (for "document"). In a supervised setting, we have a training set of N documents, each labeled with a class: $\{(d_1, c_1), \dots, (d_N, c_N)\}$. The objective is to train a classifier that can accurately map a new document d to its correct class c, where C represents a set of relevant document classes. A probabilistic classifier not only predicts the class but also provides the probability of the document belonging to each class. This full probability distribution can be valuable for making more informed downstream decisions, as it allows us to delay making discrete decisions, which can be beneficial when integrating multiple systems.

Many kinds of machine learning algorithms are used to build classifiers. This blog we are going to use Naive Bayes it belongs to a family of **Generative classifiers**

It is called Naive Bayes beacuse it is a bayesian classifier that makes a naive assumption that the features are independent of each other

For this blog we are going to use Twitter sentiment analysis dataset from [Kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

Before implementing the Naive Bayes classifier, let's first understand the theory behind it.

Naive Bayes is a probabilistic classifier, meaning that for a document $d$, out of all classes $c \in C$ the classifier returns the class $\hat{c}$ which has the maximum posterior probability, posterior probability is the probability of the class given the document $P(c \mid d)$, which can be calculated using Bayes' theorem:

$$P(c \mid d) = \frac{P(d \mid c)P(c)}{P(d)}$$

where:
$P(c \mid d)$ : Posterior probability of class $c$ given document $d$
$P(d \mid c)$ : Likelihood of document $d$ given class $c$
$P(c)$ : Prior probability of class $c$
$P(d)$ : Probability of document $d$ (usually it is called evidence or marginal likelihood)

The denominator $P(d)$ is a constant for all classes, so we can ignore it while calculating the maximum posterior probability. The class $\hat{c}$ that maximizes the posterior probability can be calculated as:

$$\hat{c} = \arg\max_{c \in C} P(c \mid d)$$

Substituting the Bayes' theorem in the above equation, we get:

$$\hat{c} = \arg\max_{c \in C} P(c \mid d) =  \arg\max_{c \in C} \frac{P(d \mid c)P(c)}{P(d)}$$

the denominator $P(d)$ is a constant for all classes, we are always asking for most likely class $c$ given document $d$ so we can ignore it, so the equation becomes:

$$\hat{c} = \arg\max_{c \in C} P(d \mid c)P(c)$$

The likelihood $P(d \mid c)$ is the probability of observing document $d$ given class $c$. In the context of text classification, we can calculate it as the product of the probabilities of observing each word in the document given the class. This is where the "naive" assumption comes into play: we assume that the features (words) are conditionally independent given the class. This assumption simplifies the calculation of the likelihood, as we can calculate the probability of observing each word independently and multiply them together. The likelihood can be calculated as:

$$P(d \mid c) = P(w_1, w_2, \dots, w_n \mid c) = \prod_{i=1}^{n} P(w_i \mid c)$$

where:
$P(w_i \mid c)$ : Probability of observing word $w_i$ given class $c$

The prior $P(c)$ is the probability of class $c$ occurring in the training set. It can be calculated as the fraction of documents in the training set that belong to class $c$:

$$P(c) = \frac{N_c}{N}$$

where:
$N_c$ : Number of documents in the training set that belong to class $c$
$N$ : Total number of documents in the training set

Now that we have the likelihood and the prior, we can calculate the posterior probability of each class for a given document. The class with the maximum posterior probability is the predicted class for the document.

To apply the naive Bayes classifier to text, we will use each word in the documents as a feature, as suggested above, and we consider each of the words in the document by walking an index through every word position in the document

$$ c_{NB} = \arg\max_{c \in C} P(c) \prod_{i  = 1}^{n} P(w_i \mid c)$$

where:
$ i \in \text{positions}$: is the index of the word in the document
$c_{NB}$ : Predicted class for the document using the naive Bayes classifier
$P(c)$ : Prior probability of class $c$
$P(w_i \mid c)$ : Probability of observing word $w_i$ given class $c$

To calculate the probability of $P(w_i \mid c)$, we’ll assume a feature is just the existence of a word in the document’s bag of words, and so we’ll want $P(w_i  \mid c)$, which we compute as the fraction of times the word $w_i$ appears among all words in all documents of topic $c$. we first concatenate all the documents of class $c$ into one big document, and then we calculate the frequency of each word in the big document to give maximum likelihood estimate of the probability. The probability of observing word $w_i$ given class $c$ can be calculated as:

$$P(w_i \mid c) = \frac{count(w_i, c)}{\sum_{w \in V} count(w, c)}$$

where:
$count(w_i, c)$ : Number of times word $w_i$ appears in documents of class $c$
$\sum_{w \in V} count(w, c)$ : Total number of words in documents of class $c$

But the above equation can be problematic when we encounter a word that is not present in the training set.
For example, Imagine we are trying to classify a document that contains the word "apple", but the word "apple" was not present in the training set. In this case, the probability $P(w_i \mid c)$ will be zero, which will make the entire likelihood zero. To avoid this problem, we can use Laplace smoothing, which adds a small constant $\alpha$ to the numerator and denominator of the probability calculation:

$$P(w_i \mid c) = \frac{count(w_i, c) + \alpha}{\sum_{w \in V} count(w, c) + \alpha \mid V \mid }$$

where:
$\alpha$ : Smoothing parameter
$ \mid V \mid $ : Size of the vocabulary (total number of unique words in the training set)

Let's walkthrough a simple example to understand how the Naive Bayes classifier works. Consider a training set with two classes: positive and negative. The training set contains the following documents:

| Document | Class |
|----------|-------|
| I love this movie | positive |
| Just plain boring | negative |
| most fun movie ever | positive |
| no fun at all | negative |

vocabulary for positive class: {I, love, this, movie, most, fun, ever}
vocabulary for negative class: {just, plain, boring, no, fun, at, all}
complete vocabulary: {I, love, this, movie, most, fun, ever, just, plain, boring, no, at, all}

Given the above training set, we want to classify the document "I had fun". To do this, we first calculate the prior probabilities of each class:

$P(\text{positive}) = \frac{\text{Number of positive documents}}{\text{Total number of documents}} = \frac{2}{4} = 0.5$

$P(\text{negative}) = \frac{\text{Number of negative documents}}{\text{Total number of documents}} = \frac{2}{4} = 0.5$

Next, we calculate the likelihood of observing the document "I had fun" given each class. We calculate the likelihood for each word in the document and multiply them together:

For the positive class:

$P(I \mid positive) = \frac{\text{I in positive documents} + \alpha}{\text{Total words in positive documents} + \alpha \mid \text{Vocabulary} \mid } = \frac{1 + 1}{7 + 1*12} = \frac{2}{19}$

$P(had \mid positive) = \frac{\text{had in positive documents} + \alpha}{\text{Total words in positive documents} + \alpha \mid \text{Vocabulary} \mid } = \frac{0 + 1}{7 + 1*12} = \frac{1}{19}$

$P(fun \mid positive) = \frac{\text{fun in positive documents} + \alpha}{\text{Total words in positive documents} + \alpha \mid \text{Vocabulary} \mid } = \frac{2 + 1}{7 + 1*12} = \frac{3}{19}$

For the negative class:

$P(I \mid negative) = \frac{\text{I in negative documents} + \alpha}{\text{Total words in negative documents} + \alpha \mid \text{Vocabulary} \mid } = \frac{0 + 1}{5 + 1*12} = \frac{1}{17}$

$P(had \mid negative) = \frac{\text{had in negative documents} + \alpha}{\text{Total words in negative documents} + \alpha \mid \text{Vocabulary} \mid } = \frac{0 + 1}{5 + 1*12} = \frac{1}{17}$

$P(fun \mid negative) = \frac{\text{fun in negative documents} + \alpha}{\text{Total words in negative documents} + \alpha \mid \text{Vocabulary} \mid } = \frac{1 + 1}{5 + 1*12} = \frac{2}{17}$

Finally, we calculate the posterior probability of each class for the document "I had fun" and predict the class with the maximum posterior probability:

$P(positive \mid I had fun) = P(I \mid +) \times P(had \mid +) \times P(fun \mid +) \times P(positive) = \frac{2}{19} \times \frac{1}{19} \times \frac{3}{19} \times 0.5 = 0.0006$

$P(negative \mid I had fun) = P(I \mid -) \times P(had \mid -) \times P(fun \mid -) \times P(negative) = \frac{1}{17} \times \frac{1}{17} \times \frac{2}{17} \times 0.5 = 0.0003$

Since $P(positive \mid I had fun) > P(negative \mid I had fun)$, we predict the document "I had fun" as belonging to the positive class.

Now that we have a good understanding of the theory behind the Naive Bayes classifier, let's implement it in Python using the Twitter sentiment analysis dataset.

First we need to load the dataset and preprocess it. We will also split the dataset into training and testing sets.

```python
data = pd.read_csv('twitter_training.csv',header=None,usecols=[2,3],names=['sentiment','text'])
data = data.sample(frac=0.4).reset_index(drop=True)
data.dropna(inplace=True)
```

Next, we need to preprocess the text data by tokenizing the text and removing stopwords. and then we will convert the text data into a matrix of token counts using the CountVectorizer class from scikit-learn.(It is an implementation of the bag of words model)

```python
def tokenize(text):
    return [word for word in word_tokenize(text) if word.isalpha() and word not in stop_words]

vectorizer = CountVectorizer(tokenizer=tokenize)
X = vectorizer.fit_transform(data.text).toarray()
y = (data.sentiment == 'Positive').astype(int)
```

Next, we will split the data into training and testing sets using the train_test_split function from scikit-learn.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Now we can train the Naive Bayes classifier using the training data. We will use the MultinomialNB class from scikit-learn to train the classifier.

```python
nb = MultinomialNB()
nb.fit(X_train, y_train)
```

Finally, we can evaluate the performance of the classifier on the testing data using the score method.

```python
nb.score(X_test, y_test)
```

The score method returns the mean accuracy of the classifier on the testing data. In this case, the accuracy represents the proportion of correctly classified documents in the testing set.

Although Accuracy is a good metric to evaluate the performance of the classifier, it is not always the best metric, especially when the classes are imbalanced. Other metrics such as precision, recall, and F1 score can provide more insights into the performance of the classifier.

Precision is the proportion of true positive predictions among all positive predictions:

$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$

Recall is the proportion of true positive predictions among all actual positive instances:

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

The F1 score is the harmonic mean of precision and recall:

$$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

We can calculate these metrics using the precision_score, recall_score, and f1_score functions from scikit-learn.

```python
print(f'Accuracy: {accuracy_score(y_test, nb.predict(X_test)):0.2f}')
print(f'F1: {f1_score(y_test, nb.predict(X_test)):0.2f}')
print(f'Precision: {precision_score(y_test, nb.predict(X_test)):0.2f}')
print(f'Recall: {recall_score(y_test, nb.predict(X_test)):0.2f}')
```

we get the following output:

```plaintext
Accuracy: 0.85
F1: 0.84
Precision: 0.84
Recall: 0.83
```

Before concluding, I want to highlight one more important aspect of Naive Bayes: it is a generative classifier. This means that it models the joint probability distribution of the features and the class, allowing it to generate new samples from the learned distribution. This capability can be particularly useful in scenarios where we need to create new samples that resemble the training data.

let's generate a new sample using the learned naive bayes model.

```python
import numpy as np
generated_class = 1
generated_features = np.random.multinomial(n=1, pvals=np.exp(nb.feature_log_prob_[generated_class]))

generated_data_point = vectorizer.inverse_transform(generated_features.reshape(1, -1))

print("Generated Data Point:", generated_data_point[0][0])
```

output:

```plaintext
Generated Data Point: interesting
```

In this blog, we explored the theory behind the Naive Bayes classifier and implemented it in Python using the Twitter sentiment analysis dataset. We also evaluated the performance of the classifier using accuracy, precision, recall, and F1 score. Finally, we generated a new sample using the learned Naive Bayes model.
