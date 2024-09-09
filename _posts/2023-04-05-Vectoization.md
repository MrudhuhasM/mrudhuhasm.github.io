---
title: "Vectorization: Transforming Text into Numbers for Machine Learning "
date: 2023-04-05
categories: [Natural_Language_Processing, Vectorization]
tags: [NLP, Vectorization, Word_Embeddings, TF-IDF, BOW]
comments: false
math: true
---

When we talk about language, words carry meaning, but to a computer, words are just gibberish. How do we make computers understand text? The answer lies in **vectorization**—the magical process of transforming words, sentences, or even entire documents into numbers that machine learning models can process. Vectorization is a critical bridge between human language and machine learning.

In this post, we’ll take a journey through the various vectorization techniques used in NLP, including one-hot encoding, Bag of Words (BOW), and TF-IDF. Let’s dive in and explore how we can convert text into numbers that power everything from spam filters to recommendation systems!

---

## The Challenge: Why Do We Need Vectorization?  

Imagine trying to teach a computer to recognize spam emails. You can't just feed it the raw email text—it would have no idea what to do with it! What we need is a way to turn text into a format that machine learning models can digest—**numbers**. That’s where vectorization comes in. It converts words into vectors (arrays of numbers) that a machine learning algorithm can understand and learn from.

Let’s say we have these three simple sentences:

1. "I love NLP"
2. "I hate NLP"
3. "I enjoy NLP"

Our task is to convert these sentences into vectors that capture their essence. This way, a model can easily process them, spot patterns, and learn from the data.

---

## One-Hot Encoding: The Basic Building Block

Let’s start with **one-hot encoding**, the simplest and most intuitive way to represent words. Each word in our vocabulary gets its own unique vector. In this case, our vocabulary is `["I", "love", "hate", "enjoy", "NLP"]`. For each word, we create a vector that has a `1` in the position of the word and `0` elsewhere. Here’s what that looks like:

- "I": `[1, 0, 0, 0, 0]`
- "love": `[0, 1, 0, 0, 0]`
- "hate": `[0, 0, 1, 0, 0]`
- "enjoy": `[0, 0, 0, 1, 0]`
- "NLP": `[0, 0, 0, 0, 1]`

We can represent each sentence as a collection of these one-hot vectors:

```python
# Sentence: "I love NLP"
[[1, 0, 0, 0, 0],  # "I"
 [0, 1, 0, 0, 0],  # "love"
 [0, 0, 0, 0, 1]]  # "NLP"
```

While one-hot encoding is easy to understand, it has **limitations**. First, the vectors can get **huge** for large vocabularies. If you have 10,000 words, each vector will be 10,000 elements long! Worse, this method doesn’t capture any relationships between words. In one-hot encoding, “love” and “hate” are as unrelated as “apple” and “neutron star,” even though they are opposites.

---

## Bag of Words (BOW): Counting Word Occurrences  

A more sophisticated approach is the **Bag of Words (BOW)** model. Here, we ignore word order and grammar, focusing only on word frequency. Each document (sentence, paragraph, or entire text) is represented by a vector that counts the number of times each word in the vocabulary appears.

Here’s how it looks with our three sentences:

```python
from sklearn.feature_extraction.text import CountVectorizer

sentences = [
    "I loved the movie",
    "Movie was awesome, movie characters were awesome",
    "Movie was terrible"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

**Output:**

```python
['awesome', 'characters', 'loved', 'movie', 'terrible', 'the', 'was', 'were']
[[0 0 1 1 0 1 0 0]
 [2 1 0 2 0 0 1 1]
 [0 0 0 1 1 0 1 0]]
```

In this example, the words `"movie"`, `"awesome"`, and `"was"` show up multiple times, and each sentence is converted into a vector of word counts.

**Limitations of BOW**: While BOW captures word frequency, it ignores the word order and context. For instance, “I hate NLP” and “I love NLP” would appear quite similar in a BOW representation, even though they have opposite meanings!

---

## TF-IDF: Adding Weight to Words  

Enter **TF-IDF (Term Frequency-Inverse Document Frequency)**, a more advanced vectorization technique that weighs words by how important they are in the document, compared to how common they are in the entire dataset.

The **TF** part measures how often a word appears in a document, while **IDF** measures how rare the word is across all documents. Words that are common across many documents (like "the" or "is") get lower scores, while unique words get higher scores.

Term Frequency (TF) is calculated as follows:

$$TF(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

where:

- $f_{t,d}$ is the frequency of term $t$ in document $d$.
- $\sum_{t' \in d} f_{t',d}$ is the total number of terms in document $d$.

Inverse Document Frequency (IDF) is calculated as follows:

$$IDF(t, D) = \log\left(\frac{N}{n_t}\right)$$

where:

- $N$ is the total number of documents.
- $n_t$ is the number of documents containing term $t$.

The final TF-IDF score is the product of TF and IDF:

$$TFIDF(t, d, D) = TF(t, d) \times IDF(t, D)$$

Here’s how TF-IDF works in action:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

sentences = [
    "I love NLP",
    "I hate NLP",
    "I enjoy NLP"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

**Output:**

```python
['enjoy', 'hate', 'love', 'nlp']
[[0.         0.         0.861037   0.50854232]
 [0.         0.861037   0.         0.50854232]
 [0.861037   0.         0.         0.50854232]]
```

As you can see, TF-IDF gives higher importance to the unique words ("love," "hate," "enjoy") while giving less weight to common words like "NLP" that appear in all three sentences.

---

## What About Word Embeddings?  

So far, we've been dealing with sparse, high-dimensional vectors that don’t capture relationships between words. But what if we want vectors that understand word **meaning**? That’s where **word embeddings** like **Word2Vec**, **GloVe**, and **BERT** come in.

Word embeddings create dense vectors, where words with similar meanings are close to each other in vector space. For example, the words "king" and "queen" might be close, and “man” and “woman” might also be near each other.

Unlike one-hot encoding or BOW, word embeddings can understand that words like “dog” and “puppy” are more similar than “dog” and “car.” These dense vectors are incredibly powerful for tasks like text classification, sentiment analysis, and even machine translation.

---

## Use Case: Finding Similar Documents  

Vectorization isn’t just for building models—it’s also handy for finding similar documents. Imagine you’re running a recommendation system and want to suggest articles that are similar to one a user has read.

We can use **cosine similarity** to compare the vectors of documents and find the most similar ones. Here’s an example of how we can use cosine similarity with BOW:

```python
from sklearn.metrics.pairwise import cosine_similarity

sentences = [
    "Cat is drinking milk",
    "Dog is running behind the cat",
    "A man is riding a horse",
    "Cat is playing with the mouse",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

new_doc = "Cat and mouse are playing"
Y = vectorizer.transform([new_doc])

cs = cosine_similarity(X, Y)

sorted_index = np.argsort(cs, axis=0)[::-1].flatten()
for idx in sorted_index:
    print(sentences[idx], cs[idx])
```

**Output:**

```python
Cat is playing with the mouse [0.70710678]
Cat is drinking milk [0.28867513]
Dog is running behind the cat [0.23570226]
A man is riding a horse [0.]
```

The output shows how similar the new document is to existing ones, based on their BOW vector representation. This is the backbone of document recommendation systems!

---

## Wrapping It Up: From Sparse to Dense Vectors  

In this blog, we explored how vectorization transforms words into numbers through different techniques. **One-hot encoding**, **Bag of Words**, and **TF-IDF** are all powerful methods for turning text into data that machine learning models can process. While these methods have limitations (like not capturing word meaning), they are great for many text processing tasks.

If you want your model to understand word semantics, **word embeddings** are the next step in NLP, offering dense representations that carry rich information about words and their relationships.

Whatever your text-processing task—whether it's document classification, sentiment analysis, or building a search engine—vectorization is a critical tool in your NLP toolkit.

---



**How do you approach vectorization in your NLP projects? Do you prefer TF-IDF or word embeddings? Share your thoughts and experiences in the comments below!**