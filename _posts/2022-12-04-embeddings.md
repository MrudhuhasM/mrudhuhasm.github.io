---
title: "Embeddings"
date: 2022-12-04
categories: [Natural_Language_Processing, Embeddings]
tags: [machine-learning, embeddings, natural-language-processing]
comments: false
math: true
---

The history of Language AI is filled with notable milestones, all united by the goal of enabling machines to effectively understand and generate human language. Over the decades, there have been remarkable advancements in algorithms, neural networks, and computational linguistics, all working towards bridging the gap between human communication and machine comprehension. However, human language remains inherently complex and nuanced, presenting unique challenges for traditional computational methods.

Language is, by its nature, ambiguous and deeply contextual. It doesn't adhere to a fixed structure that machines can easily interpret. Words can have multiple meanings depending on the context, and a sentence may convey different ideas based on cultural or situational factors. The subtleties of language—metaphors, idioms, slang, and tone—are particularly difficult for machines to grasp. When language is translated into data, often in the form of binary code, much of its richness and subtlety can be lost.

**Vector Semantics** offers a solution by representing words in a continuous vector space. The concept is to map words as vectors in a high-dimensional space, where each dimension corresponds to a feature of the word. The key idea is that words with similar meanings should have similar vector representations. 

These word vectors are commonly referred to as **embeddings**. Embeddings are dense vector representations learned from large text corpora using unsupervised learning methods. They capture the semantic meaning of words and are widely used in various natural language processing tasks, including sentiment analysis, machine translation, and named entity recognition.

One of the earliest and simplest models in NLP is the **Bag of Words (BOW)** model. The BOW approach converts text into numerical form, enabling it to be used in machine learning algorithms. It breaks down a sentence or document into individual words, disregarding grammar and word order, and treats each word as an independent feature. A vector is created based on the frequency or presence of these words. In this model, text is treated as an unordered "bag" of words, hence the name.

While the Bag of Words model is straightforward and easy to implement, it has its limitations. Since it ignores word order, it often misses the semantic meaning embedded in the text. For example, the sentences "The dog chased the cat" and "The cat chased the dog" would be represented identically, even though their meanings are quite different. Additionally, BOW struggles to capture the context in which words are used. Despite these drawbacks, the model provided a foundational stepping stone for more sophisticated language models that followed.

Let's walk through an example of how to implement the Bag of Words model in Python using the `scikit-learn` library:

```python
from sklearn.feature_extraction.text import CountVectorizer

# List of text documents
text = ["The quick brown fox jumped over the lazy dog.",
        "The dog barked at the fox.",
        "The fox ran away quickly.",
        "The quick brown dog jumped over the lazy fox."]

# Create the transform
vectorizer = CountVectorizer(lowercase=True)

# Tokenize and build vocab
vectorizer.fit(text)
print(vectorizer.vocabulary_)
```

The output will be a dictionary where the keys are the words from the text and the values are the indices assigned to them, representing the vocabulary.

```python
{'the': 12, 'quick': 9, 'brown': 3, 'fox': 5, 'jumped': 6, 'over': 8, 'lazy': 7, 'dog': 4, 'barked': 2, 'at': 0, 'ran': 11, 'away': 1, 'quickly': 10}
```

You can also retrieve the vocabulary words using:

```python
print(vectorizer.get_feature_names_out())
```

```python
['at' 'away' 'barked' 'brown' 'dog' 'fox' 'jumped' 'lazy' 'over' 'quick' 'quickly' 'ran' 'the']
```

When we transform the text data using the `transform` method, the output is a matrix where each row corresponds to a document, and each column corresponds to a word from the vocabulary. The values represent the frequency of each word in the respective document.

```python
[[0 0 0 1 1 1 1 1 1 1 0 0 2]
 [1 0 1 0 1 1 0 0 0 0 0 0 2]
 [0 1 0 0 0 1 0 0 0 0 1 1 1]
 [0 0 0 1 1 1 1 1 1 1 0 0 2]]
```

If you compare the first and last rows, you’ll notice that the Bag of Words model treats both sentences as identical vectors, despite their meanings being quite different. This highlights one of the model’s key limitations: it cannot capture word order or context.

**Raw frequency counts** in the BOW model can be problematic because they don't account for the relative importance of words within a document. Two common approaches to address this issue are:

1. **Term Frequency-Inverse Document Frequency (TF-IDF):** This statistical measure evaluates the importance of a word in a document relative to a collection of documents. TF-IDF is calculated as the product of two terms: Term Frequency (TF) and Inverse Document Frequency (IDF). The TF term measures how frequently a word appears in a document, while IDF measures how rare the word is across the entire collection. This helps highlight important words that are frequent in a specific document but uncommon in the overall corpus.

   $$TF-IDF(w,d,D) = TF(w,d) \times IDF(w,D)$$

   Where:
   - $TF(w,d)$ is the term frequency of word $w$ in document $d$
   - $IDF(w,D)$ is the inverse document frequency of word $w$ in the collection of documents $D$

   TF-IDF is widely used because it helps capture the relative importance of words. However, it still lacks the ability to capture the full semantic meaning of words.

2. **Positive Pointwise Mutual Information (PPMI):** PPMI measures the association between two words in a corpus, based on how often they co-occur. It is calculated as the pointwise mutual information (PMI) between two words, with negative values set to zero. Higher PPMI values indicate a stronger association, making it a useful method for constructing word embeddings that capture semantic meaning.

   $$PPMI(w,c) = \max\left(\log \frac{P(w,c)}{P(w)P(c)}, 0\right)$$

   Where:
   - $P(w,c)$ is the joint probability of words $w$ and $c$
   - $P(w)$ and $P(c)$ are the marginal probabilities of $w$ and $c$, respectively.

By using these methods, we can better address the shortcomings of simpler models like BOW and create more powerful representations of text.

### From Word2Vec to Modern Embedding Models

While methods like **Bag of Words** (BOW), **TF-IDF**, and **PPMI** provide foundational ways to represent text, they fail to capture the rich semantic relationships between words. These models don't account for the context in which words appear, nor do they offer a way to understand the similarity between words beyond their simple co-occurrence. 

This is where **Word Embeddings** revolutionized the field of Natural Language Processing (NLP). Embeddings are dense, continuous vector representations of words or phrases that capture their semantic meaning by positioning them in a high-dimensional vector space. Words that are similar in meaning appear closer together in this space. Embeddings can be learned from large corpora of text and are now a fundamental building block in modern NLP.

### Word2Vec: A Pioneer in Embedding Models

The release of **Word2Vec** by Google in 2013 was a significant milestone in NLP. It provided a method to represent words as dense vectors, where the position of a word in the vector space is determined by the surrounding words in the corpus (context). Word2Vec uses two main architectures to learn these embeddings:

1. **Continuous Bag of Words (CBOW):** The model predicts the target word based on the context (surrounding words).
   
   - Example: In the sentence "The quick brown fox jumps over the lazy dog," given the context "The quick brown ___ jumps over," the model predicts the missing word "fox."

2. **Skip-Gram Model:** This model does the inverse, predicting the context words given the target word.
   
   - Example: Given the word "fox," the model predicts "quick," "brown," "jumps," etc., as its surrounding context.

Word2Vec learns embeddings that allow words with similar contexts to have similar vector representations. This was a breakthrough because it captured both the meaning of words and their relationships, making it possible to perform tasks such as analogy solving (e.g., "king" - "man" + "woman" ≈ "queen").

#### Example of Word2Vec in Python

Let’s look at how you can implement **Word2Vec** using the `gensim` library in Python:

```python
from gensim.models import Word2Vec

# Example sentences (tokenized)
sentences = [
    ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
    ['the', 'dog', 'barked', 'at', 'the', 'fox'],
    ['the', 'fox', 'ran', 'away', 'quickly'],
    ['the', 'quick', 'brown', 'dog', 'jumped', 'over', 'the', 'lazy', 'fox']
]

# Train a Word2Vec model using the skip-gram method
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

# Get the word vector for "fox"
vector_fox = model.wv['fox']
print(vector_fox)
```

This code trains a Word2Vec model on a set of sentences and outputs the vector representation of the word "fox." Word vectors like this can then be used in downstream tasks such as sentiment analysis, document classification, or named entity recognition.

### Pre-trained Embeddings: GloVe and FastText

Following Word2Vec, other methods were introduced to enhance word embeddings. Two notable models include **GloVe** and **FastText**.

#### GloVe: Global Vectors for Word Representation

**GloVe** (Global Vectors for Word Representation) is an unsupervised learning algorithm for obtaining vector representations for words. Unlike Word2Vec, which focuses on local context windows, GloVe uses **global word co-occurrence statistics** to learn word embeddings. The intuition behind GloVe is that ratios of word co-occurrences carry important information about word meaning.

In GloVe, a word co-occurrence matrix is constructed, and a factorization technique is applied to learn the embeddings. GloVe embeddings are widely used because they balance the benefits of global corpus statistics (captured by methods like TF-IDF) with the semantic power of dense vectors.

You can load and use pre-trained GloVe embeddings directly from sources like Stanford NLP, which provide embeddings trained on large corpora like Wikipedia and Common Crawl:

```python
import numpy as np

# Load pre-trained GloVe embeddings (example with 100-dimensional vectors)
embedding_dict = {}
with open("glove.6B.100d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_dict[word] = vector

# Retrieve the vector for the word "king"
vector_king = embedding_dict.get("king")
print(vector_king)
```

#### FastText: Subword Information

**FastText**, developed by Facebook, extends Word2Vec by addressing one of its key limitations: its inability to represent out-of-vocabulary words. FastText creates embeddings for subword units (character n-grams), which allows it to generate vectors for words that were not seen during training.

This is particularly useful for languages with rich morphology or for tasks where spelling errors or rare words are common.

```python
from gensim.models import FastText

# Example sentences (tokenized)
sentences = [
    ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
    ['the', 'dog', 'barked', 'at', 'the', 'fox'],
    ['the', 'fox', 'ran', 'away', 'quickly'],
    ['the', 'quick', 'brown', 'dog', 'jumped', 'over', 'the', 'lazy', 'fox']
]

# Train a FastText model
model = FastText(sentences, vector_size=100, window=5, min_count=1)

# Get vector for an unseen word, e.g., "foxed" (assuming it's not in the training set)
vector_foxed = model.wv['foxed']
print(vector_foxed)
```

### Embeddings from Transformer Models

Recent advancements in NLP have brought about **contextual embeddings** from transformer-based models like BERT, GPT, and their derivatives. Unlike static word embeddings (Word2Vec, GloVe, FastText), these embeddings are **context-dependent**—the same word will have different embeddings based on the surrounding words.

#### BERT: Bidirectional Encoder Representations from Transformers

BERT (Bidirectional Encoder Representations from Transformers) generates embeddings by looking at both the left and right context of a word. This allows BERT to create highly nuanced, context-aware embeddings.

With libraries like Hugging Face’s `transformers`, you can easily use BERT for generating word or sentence embeddings:

```python
from transformers import BertModel, BertTokenizer
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example sentence
sentence = "The quick brown fox jumps over the lazy dog."

# Tokenize the sentence
inputs = tokenizer(sentence, return_tensors="pt")

# Generate BERT embeddings
with torch.no_grad():
    outputs = model(**inputs)

# The final hidden states for each token in the sentence
token_embeddings = outputs.last_hidden_state
print(token_embeddings)
```

### How to Use Embeddings in Deep Learning Tasks

Embeddings can be applied to a wide variety of downstream tasks, including:

1. **Text Classification**: Embeddings can be fed into models like Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), or fully connected layers for tasks like sentiment analysis or document classification.
   
   - **Example**: In sentiment analysis, sentence embeddings can be passed to an RNN or Transformer to classify whether the sentiment is positive or negative.

2. **Named Entity Recognition (NER)**: Using contextual embeddings from models like BERT, we can train models to identify and classify entities (e.g., names, organizations) within a text.
   
3. **Text Similarity**: Word and sentence embeddings can be used to calculate similarity scores between texts, useful for information retrieval or document clustering.
   
   - **Example**: Use cosine similarity between sentence embeddings to rank search results based on relevance to a query.

4. **Machine Translation**: In translation systems, embeddings capture the semantic meaning of words and help models align the meaning of words and phrases across languages.
   
5. **Question Answering**: Contextual embeddings from transformer models can be used to understand both the question and the context, enabling more accurate answers.

### Conclusion

Word embeddings have come a long way since the early days of Bag of Words models. From **Word2Vec** to **GloVe** and **FastText**, and now to modern transformer-based models like **BERT**, embeddings are essential for representing the meaning of words and sentences in NLP tasks. As models continue to evolve, so too will the sophistication with which machines can understand and generate human language.

By leveraging embeddings, we can power a wide range of applications, from simple text classification tasks to complex question answering and machine translation systems. The ability to capture and utilize semantic meaning at scale is what makes embeddings such a powerful tool in modern AI.

