---
title: "Sentiment Analysis with Logistic Regression"
date: 2022-09-24
categories: [Natural_Language_Processing, Text_Classification]
tags: [machine-learning, sentiment-analysis, logistic-regression,nlp]
comments: false
math: true
---

In previous posts, we explored sentiment analysis using a simple bag-of-words model and Naive Bayes. In this post, we will delve into sentiment analysis with logistic regression. Unlike Naive Bayes, which is a generative model, logistic regression is a discriminative model. The key difference lies in their approach: a generative model estimates the joint probability of the features and the target variable, whereas a discriminative model estimates the conditional probability of the target variable given the features.

### Components of a Machine Learning Classifier

1. **Feature Representation:** A method to represent the input data.
2. **Classification Function:** A function that estimates the class $\hat{y}$ given the input features $x$ by modeling $P(y \mid x)$.
3. **Loss Function:** A measure of the discrepancy between the predicted class and the true class.
4. **Optimization Algorithm:** A method to minimize the loss function.

After acquiring the data, the text data must be tokenized and vectorized. Various vectorization techniques can be employed, such as bag-of-words, TF-IDF, or word embeddings.

In this instance, we will use subword tokenization along with basic encoding, utilizing the `Hugging Face` library for text tokenization.

For the classification function, we will implement logistic regression with a sigmoid activation function.

Let's first understand the sigmoid function:

The goal of Logistic Regression is to train a classifier that can distinguish between two classes. The sigmoid function is used to squash the input value between 0 and 1, which can be interpreted as the probability of the input belonging to a particular class.

consider a single input observation $x$ and a binary classification problem with two classes, $C_1$ and $C_2$. The sigmoid function is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

where $z$ is the linear combination of the input features $x$ and the model parameters $\theta$:

Here we want to know the probability of the input belonging to class $P(y=1\mid x)$

The decision is positive sentiment vs negative sentiment.the features represent the words in the text.

$P(y=1\mid x)$ is the probability of the input belonging to positive sentiment.
$P(y=0\mid x)$ is the probability of the input belonging to negative sentiment.

Logistic regression solves this task by learning the weights $\theta$ and bias $b$ that minimize the loss function. The loss function used in logistic regression is the binary cross-entropy loss, which measures the difference between the predicted probability and the true label.

The binary cross-entropy loss is defined as:

$$
\text{Loss} = -\frac{1}{N}\sum_{i=1}^{N}y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)
$$

where $y_i$ is the true label and $\hat{y}_i$ is the predicted probability.

To make a prediction on test instance after training the classifier first multiply the input features with the learned weights and bias. The reulting value is weighted sum of the input features.

$$
z = \theta^Tx + b
$$

But here $z$ can be any value, we need to squash it between 0 and 1. This is done by passing the weighted sum through the sigmoid function.

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

The output of the sigmoid function is the predicted probability of the input belonging to class 1.

To make output of sigmoid a probability, we need to make sure the two cases $P(y=1\mid x) + P(y=0\mid x) = 1$ which is why we have $P(y=0\mid x) = 1 - P(y=1\mid x)$

Once we have the predicted probability, we can make a decision by setting a threshold. If the predicted probability is greater than the threshold, we predict class 1; otherwise, we predict class 0.The threshold is also called the decision boundary.

Let's walk through a simple example to understand the logistic regression model. Consider a dataset with two features, $x_1$ and $x_2$, and a binary target variable, $y$. The goal is to predict the class of the input based on the features.

| $x_1$ | $x_2$ | $y$ |
|-------|-------|-----|
| 1     | 2     | 0   |
| 2     | 3     | 0   |
| 3     | 4     | 1   |
| 4     | 5     | 1   |

The logistic regression model is defined as:

$$
\hat{y} = \sigma(\theta_1x_1 + \theta_2x_2 + b)
$$

where $\theta_1$, $\theta_2$, and $b$ are the model parameters.

Let's assume the model parameters are $\theta_1 = 0.5$, $\theta_2 = -0.2$ and $b = 1$. We can calculate the predicted probability for a test instance with $x_1 = 2$ and $x_2 = 2$ as follows:

$$
z = 0.5 \times 2 - 0.2 \times 2 + 1 = 1.6
$$

$$
\hat{y} = \sigma(1.6) = \frac{1}{1 + e^{-1.6}} = 0.832
$$

The predicted probability is 0.832, which is greater than the threshold of 0.5. Therefore, we predict class 1 for the test instance.

### Naive Bayes vs Logistic Regression

Naive Bayes has a strong assumption of feature independence, which may not hold true in practice. Logistic regression does not make this assumption and can capture complex relationships between features.

Consider two features, $x_1$ and $x_2$, that are highly correlated. Naive Bayes would treat these features as independent and multiply their probabilities, which may lead to incorrect predictions. Logistic regression, on the other hand, can model the relationship between these features and make more accurate predictions.

### Multinomial Logistic Regression

In the case of multi-class classification, we can extend logistic regression to the multinomial logistic regression model.

The softmax function is used to compute the probability of each class given the input features.

The softmax function is defined as:

$$
\text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
$$

where $z$ is the linear combination of the input features and the model parameters, and $K$ is the number of classes.

The loss function used in multinomial logistic regression is the cross-entropy loss, which measures the difference between the predicted probabilities and the true labels.

### Cross-Entropy Loss

We need a loss function that expresses, for an observation $x$, how close the classifier output $\hat{y}$ is to the true label $y$. The cross-entropy loss is a common loss function used in classification tasks.

The cross-entropy loss is defined as:

$$
\text{Loss} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}y_{ik}\log(\hat{y}_{ik})
$$

where $y_{ik}$ is the true label for observation $i$ and class $k$, and $\hat{y}_{ik}$ is the predicted probability for observation $i$ and class $k$.

For optimization, we can use gradient descent to minimize the loss function and learn the model parameters.

Gradient descent is an iterative optimization algorithm that updates the model parameters in the direction of the steepest decrease in the loss function.

Model parameters are updated as follows:

$$
\theta = \theta - \alpha \nabla_{\theta}\text{Loss}
$$

where $\alpha$ is the learning rate, and $\nabla_{\theta}\text{Loss}$ is the gradient of the loss function with respect to the model parameters.

### Conclusion

In this post, we explored sentiment analysis with logistic regression. We discussed the components of a machine learning classifier, the sigmoid function, the binary cross-entropy loss, and the softmax function. We also compared Naive Bayes and logistic regression and introduced multinomial logistic regression for multi-class classification tasks. Finally, we discussed the cross-entropy loss and gradient descent for optimization.
