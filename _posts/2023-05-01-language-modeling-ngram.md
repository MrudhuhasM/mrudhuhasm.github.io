---
title: "Language Modeling: N-gram"
date: 2023-05-01
tags: [nlp, language-modeling, n-gram, statistics]
description: "Understanding statistical language modeling with N-grams"
math: true
---
dg-publish: true
dg-permalink: language-modeling-n-gram
description: Understanding N-gram Language modeling
---

# Language Modeling - N-gram

<div></div>

<div class="article-meta">
  <span class="article-category">Natural Language Processing</span>
  <span class="reading-time" title="Estimated read time">
    • 10 min read
  </span>
</div>

<div class="article-content">

In this blog, we'll explore the foundations of n-gram models, how they work, their strengths and limitations, and how they compare to more sophisticated approaches. Understanding these classical techniques provides valuable insights into the evolution of language modeling and why more advanced models were developed.

For example, given the sentence:

```
The quick brown fox jumps over the lazy ____
```

We might predict the next word as **"dog," "cat," or "river"** based on the context, but not function words like **"the" or "this."** 

A language model is a machine learning model that predicts the next word in a sequence and assigns probabilities to each possible word. One of the simplest types of language models is the **n-gram model**.

An **n-gram** is a sequence of **n** words:
- A **1-gram (unigram)** is a single word.
- A **2-gram (bigram)** is a sequence of two words.
- A **3-gram (trigram)** is a sequence of three words, and so on.

When we refer to an **n-gram model**, we mean a model that predicts the **$n^{th}$ word given the previous $n-1$ words.** These models provide a simple yet effective way to estimate probabilities in a text sequence and serve as a foundation for more advanced techniques.

## **How Do N-gram Models Work?**

To understand n-gram models, let's begin with the fundamental task of computing the probability of a word $w$ given some history $h$:

$$
P(w|h) = \frac{C(w,h)}{C(h)}
$$

where:
- $C(w,h)$ is the count of occurrences of the sequence $(h, w)$ in a corpus.
- $C(h)$ is the count of the history $h$ appearing in the corpus.

For example, given the sentence:

```
The quick brown fox jumps over the lazy ____
```

we want to compute the probability of the word **"dog"** appearing next:

$$
P(\text{"dog"}|\text{"the quick brown fox jumps over the lazy"}) = \frac{C(\text{"the quick brown fox jumps over the lazy dog"})}{C(\text{"the quick brown fox jumps over the lazy"})}
$$

If we had a large enough dataset, we could estimate these probabilities by counting occurrences. However, natural language is highly diverse, and many word combinations might rarely appear in training data, making it difficult to estimate probabilities directly. To address this, we need a more practical way to approximate probabilities.
### **The Markov Assumption and N-gram Approximation**
This is where **n-gram models** come in. They simplify the problem by making a key assumption: the probability of a word depends only on the previous $n-1$ words, rather than the entire history. This is known as the **Markov assumption**:

$$
P(w|h) \approx P(w|h_{n-1}, h_{n-2}, ..., h_{1})
$$

This assumption drastically reduces the complexity of computing probabilities, making it feasible to estimate them using limited data. 

For example, instead of:

$$
P(\text{"dog"}|\text{"the quick brown fox jumps over the lazy"})
$$

we approximate it as:

$$
P(\text{"dog"}|\text{"lazy"})
$$

if we use a **bigram model** (n=2), meaning we consider only the last word. Similarly, for a **trigram model** (n=3), we would consider the last two words:

$$
P(\text{"dog"}|\text{"over the lazy"})
$$

Thus, in general, an **n-gram model** estimates probabilities as:

$$
P(w|h) \approx P(w|h_{n-1}, h_{n-2}, ..., h_{1})
$$

where:
- **Bigram model** (n=2):  
  $$
  P(w|h) \approx P(w|h_{n-1})
  $$
- **Trigram model** (n=3):  
  $$
  P(w|h) \approx P(w|h_{n-1}, h_{n-2})
  $$

By using this approximation, n-gram models allow us to compute word probabilities efficiently while capturing local word dependencies.
## **Estimating N-gram Probabilities**

Given the Markov assumption, we can estimate n-gram probabilities using **maximum likelihood estimation (MLE)**.

We get MLE estimates by counting occurrences of n-grams in a corpus and normalizing them to obtain probabilities. For example, to estimate bigram probabilities, we count the occurrences of each word pair $(h, w)$ and divide by the total count of history $h$:

$$
P(w|h) = \frac{C(h, w)}{C(h)}
$$

where:
- $C(h, w)$ is the count of the bigram $(h, w)$.
- $C(h)$ is the count of the history $h$.

### **Practical Example: Estimating a Bigram Probability**
Let's go through a practical example using **[The Verdict - Wikisource, the free online library](https://en.wikisource.org/wiki/The_Verdict)**.

#### **Loading the Corpus**
```python
import re
from collections import Counter
from nltk.util import ngrams

with open("the-verdict.txt", "r") as f:
    data = f.read()

words = data.replace("--", " ").lower().split()
len(words)
```
```
3731
```
Our corpus contains a total of **3731 words**. Now, let's implement a **bigram model**.

#### **Extracting Bigrams**
We first extract **all possible bigrams** from the text.

```python
bigrams = list(ngrams(words, 2))
bigrams[:10]
```
```
[('i', 'had'),
 ('had', 'always'),
 ('always', 'thought'),
 ('thought', 'jack'),
 ('jack', 'gisburn'),
 ('gisburn', 'rather'),
 ('rather', 'a'),
 ('a', 'cheap'),
 ('cheap', 'genius'),
 ('genius', 'though')]
```
Now, let's say we want to compute **$P(\text{"gisburn"}|\text{"jack"})$**, the probability of the word `gisburn` appearing after `jack`.

### **Step 1: Count All Bigrams Starting with 'Jack'**
We extract all bigrams where `"jack"` is the first word to find possible next words.

```python
jack_bigrams = [bigram for bigram in bigrams if bigram[0] == 'jack']
print(jack_bigrams)
```
```
[('jack', 'gisburn'), ('jack', 'gisburn!'), ('jack', 'one'), ('jack', 'himself,'), ('jack', 'himself')]
```
There are **5 bigrams** where `"jack"` is the first word.

### **Step 2: Count Occurrences of ('Jack', 'Gisburn')**
We count how many times `"gisburn"` follows `"jack"` in the corpus.

```python
jack_gisburn_count = sum(1 for bigram in jack_bigrams if bigram[1] == 'gisburn')
jack_gisburn_count
```
```
1
```
The bigram `("jack", "gisburn")` appears **once**.

### **Step 3: Compute Probability**
Finally, we compute the probability:

$$
P(\text{"gisburn"}|\text{"jack"}) = \frac{C(\text{"jack", "gisburn"})}{C(\text{"jack"})}
$$

```python
jack_count = len(jack_bigrams)
jack_gisburn_prob = jack_gisburn_count / jack_count
jack_gisburn_prob
```
```
0.2
```
Thus, the probability of **"gisburn"** appearing after **"jack"** is **0.2 (or 20%)**.

### **Relative Frequency and Maximum Likelihood Estimation (MLE)**
The ratio

$$
\frac{C(\text{"jack", "gisburn"})}{C(\text{"jack"})}
$$

is also called the **relative frequency**.

Using relative frequencies to estimate probabilities is a simple yet effective approach for computing **maximum likelihood estimation (MLE)** in n-gram models. However, directly multiplying probabilities can introduce computational challenges.

### **The Problem of Underflow and Log Probabilities**
Language models can be very large, leading to practical issues—one of which is computing probabilities. Since probabilities are small values between 0 and 1, multiplying many probabilities together results in very small numbers, leading to computational **underflow**. To mitigate this, we often use **log probabilities**, which are more numerically stable.

Instead of:

$$
p_1 \times p_2 \times p_3 \times p_4 \times p_5
$$

we compute:

$$
e^{\log(p_1) + \log(p_2) + \log(p_3) + \log(p_4) + \log(p_5)}
$$

By summing log probabilities instead of multiplying them, we avoid numerical instability while preserving the relationships between probabilities.

### **Evaluating N-gram Models**

To assess the performance of an n-gram model, we first split our data into training, development, and test sets. The model is trained on the training set, fine-tuned on the development set, and evaluated on the test set, which measures how well the model generalizes to unseen data.

Training set: Data used to learn the model parameters.

Development set (Dev set): A separate dataset used to tune hyperparameters and compare different models before final evaluation.

Test set: A held-out dataset used to assess the final model’s performance.

The development set is crucial because it helps prevent overfitting. If we tuned hyperparameters directly on the test set, we might unknowingly optimize the model for that specific dataset rather than ensuring generalization. By using a separate development set, we ensure that the test set remains an unbiased measure of the model's actual performance.

A good language model should generalize well to unseen text. A model that fits the training data perfectly but fails on new data is overfitting, making it ineffective in real-world applications. We measure the quality of an n-gram model by its performance on an unseen test corpus.

### **Perplexity**
To evaluate n-gram models, we compare how well they assign probabilities to a test set. However, using raw probabilities isn’t ideal because:
- The probability of a test set **decreases** as the text length increases.
- Comparing models based on raw probability is not straightforward.

Instead, we use **perplexity**, a function of probability that normalizes for text length.

#### **Definition of Perplexity**
The **perplexity (PP or PPL)** of a language model on a test set is:

$$
PP(W) = P(w_1, w_2, w_3, ..., w_N)^{-\frac{1}{N}}
$$

where:
- $PP(W)$ is the perplexity of the test set $W$.
- $P(w_1, w_2, w_3, ..., w_N)$ is the probability assigned to the test set.
- $N$ is the total number of words in the test set.

#### **Interpreting Perplexity**
Perplexity quantifies how uncertain or ‘surprised’ a model is when predicting text. A lower perplexity indicates:
- The model assigns higher probabilities to correct word sequences.
- The model is more confident in its predictions.

For example:
- A **perfect model** that assigns a probability of **1** to the correct test set would have a perplexity of **1**.
- A **random model** that assigns equal probability to all words would have a **high perplexity**.
- A **better-performing model** will have a **lower perplexity** than a weaker one.

Since different language models yield different perplexities for the same text, **perplexity is a useful metric for comparing models**.

### **Limitations of N-gram Models**

While n-gram models are simple and computationally efficient, they have several limitations:

**Data Sparsity**: As n increases, the number of possible n-grams grows exponentially, making it difficult to collect enough data to estimate probabilities accurately.

**Fixed Context Window**: N-gram models only consider a limited number of preceding words, ignoring long-range dependencies in language.

**Poor Generalization**: They rely on exact word matches, meaning they struggle with unseen n-grams or slight variations in phrasing.

**High Memory Usage**: Storing large n-gram counts requires significant memory, especially for higher-order models.

To address these limitations, modern language models use smoothing techniques (like Laplace and Kneser-Ney smoothing) and neural-based approaches like Recurrent Neural Networks (RNNs), LSTMs, and Transformers.

### **Conclusion**

N-gram language models represent a foundational approach to language modeling. They offer a simple and interpretable way to estimate word probabilities but struggle with long-range dependencies and data sparsity. Despite their limitations, they have played a crucial role in NLP and continue to be useful in applications like text compression, speech recognition, and information retrieval.

While deep learning models like Transformers have largely replaced n-gram models for complex NLP tasks, understanding n-grams provides a valuable perspective on the evolution of language modeling techniques. By combining classical approaches with modern advancements, we can build more efficient and effective NLP systems.

</div>