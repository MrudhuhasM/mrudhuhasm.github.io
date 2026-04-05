---
title: "Sentiment Analysis: From Bag of Words to Word Embeddings"
date: 2024-03-01
tags: [nlp, sentiment-analysis, word2vec, fasttext, word-embeddings]
description: "Comparing traditional bag-of-words with modern word embedding approaches for sentiment classification"
---
dg-permalink: sentiment-analysis-bow-to-embeddings
description: Comparing traditional bag-of-words with modern word embedding approaches for movie review sentiment classification
---

# Sentiment Analysis: From Bag of Words to Word Embeddings

Sentiment analysis requires converting text into numerical representations. Traditional approaches like Bag of Words (BOW) treat words as independent tokens, discarding order and semantic relationships. Word embeddings (Word2Vec, FastText) capture semantic similarity in dense vector spaces, potentially improving classification performance.

This analysis compares three approaches on IMDb movie reviews: BOW with TF-IDF weighting, Word2Vec embeddings, and FastText embeddings. Results show embedding methods achieve similar accuracy (88-89%) to BOW (87%) while providing semantic representations useful beyond classification.

**Kaggle Notebook:** [Movie Sentiment Analysis: BOW, Word2Vec, FastText](https://www.kaggle.com/code/mrudhuhas/movie-sentiment-analysis-bow-w2v-fasttext)

## Dataset and Preprocessing

IMDb movie reviews dataset: 40,000 samples balanced between positive and negative sentiment.

```python
data = pd.read_csv('movie.csv')
print(data.label.value_counts())
# 0 (negative): 20,019
# 1 (positive): 19,981
```

**Preprocessing Pipeline:**

```python
def clean_data(text):
    text = text.replace('br','')  # Remove HTML breaks
    doc = load_model(text)  # spaCy processing
    text = " ".join([token.lemma_ for token in doc])  # Lemmatization
    preprocess_functions = [
        to_lower,
        remove_special_character,
        remove_number,
        normalize_unicode,
        remove_punctuation,
        expand_contraction,
        remove_stopword
    ]
    preprocessed_text = preprocess_text(text, preprocess_functions)
    return preprocessed_text
```

Steps: HTML cleaning → lemmatization → lowercase → special character/number removal → contraction expansion → stopword removal.

Train/test split: 32,000 train, 8,000 test (80/20).

## Bag of Words with TF-IDF

BOW represents documents as word frequency vectors. TF-IDF weights down common words appearing across many documents:

$$\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log\frac{N}{df(t)}$$

where $N$ is total documents and $df(t)$ is document frequency of term $t$.

```python
vect = CountVectorizer(max_features=10000)
X_train_bow = vect.fit_transform(X_train)
X_test_bow = vect.transform(X_test)
```

Vocabulary limited to 10,000 most frequent terms, producing sparse 10,000-dimensional vectors.

### Multinomial Naive Bayes

Probabilistic classifier assuming feature independence:

```python
classifier = MultinomialNB()
classifier.fit(X_train_bow, y_train)
predictions = classifier.predict(X_test_bow)
```

**Performance:**
- Accuracy: 87.2%
- Precision (Negative): 0.88 | Recall: 0.87
- Precision (Positive): 0.86 | Recall: 0.88

Strong baseline with simple model. Independence assumption holds reasonably well for sentiment despite ignoring word order.

### Logistic Regression

Linear classifier with balanced class weights:

```python
logreg_classifier = LogisticRegression(class_weight='balanced', solver='liblinear')
logreg_classifier.fit(X_train_bow, y_train)
```

**Performance:**
- Accuracy: 88.0%
- Precision (Negative): 0.89 | Recall: 0.87
- Precision (Positive): 0.87 | Recall: 0.89

Slight improvement over Naive Bayes. Class balancing prevents bias toward majority class.

### Linear SVM

Maximum-margin classifier:

```python
vect = CountVectorizer(max_features=1000)  # Reduced features
svm_classifier = LinearSVC()
svm_classifier.fit(X_train_bow, y_train)
```

**Performance:**
- Accuracy: 86.6%
- Precision (Negative): 0.86 | Recall: 0.88
- Precision (Positive): 0.87 | Recall: 0.85

Comparable to Naive Bayes. Vocabulary reduction to 1,000 features limits performance but improves speed.

## Word2Vec Embeddings

Word2Vec learns dense vector representations where semantically similar words cluster together. Skip-gram variant predicts context words from target word.

```python
X_train_tokens = X_train.apply(word_tokenize)
w2v = Word2Vec(
    X_train_tokens,
    vector_size=200,
    window=5,
    min_count=2,
    workers=3,
    sg=1  # Skip-gram
)
```

**Semantic Relationships:**

```python
w2v.wv.most_similar('good')
# [('goodand', 0.73), ('serviceable', 0.71), ('halfdecent', 0.70), ...]
```

The model captures semantic similarity: "good" relates to "serviceable", "halfdecent", "soso"—all expressing quality assessment.

**Document Representation:**

```python
def text2vec(list_tokens):
    DIMENSION = 200
    features = []
    for tokens in list_tokens:
        row_feat = np.zeros(DIMENSION)
        c = 1e-5
        for token in tokens:
            if token in w2v.wv:
                row_feat += w2v.wv[token]
                c += 1
        features.append(row_feat / c)  # Mean pooling
    return features
```

Document vector = mean of constituent word vectors. Simple aggregation loses word order but captures overall semantic content.

### Logistic Regression with W2V

```python
log_reg_emb = LogisticRegression(solver='liblinear')
log_reg_emb.fit(X_train_vect, y_train)
```

**Performance:**
- Accuracy: 88.4%
- Precision (Negative): 0.88 | Recall: 0.89
- Precision (Positive): 0.89 | Recall: 0.88

Marginal improvement over BOW (88.4% vs 88.0%). Dense representations capture semantic similarity but mean pooling discards syntactic structure.

## FastText Embeddings

FastText extends Word2Vec by representing words as bags of character n-grams. This handles out-of-vocabulary words and morphological variations.

**Training:**

```python
# Format: text __label__class
train_file = './train.csv'
X_train.to_csv(train_file, header=None, index=False)

model = fasttext.train_supervised(
    input=train_file,
    lr=1.0,
    epoch=75,
    loss='hs',  # Hierarchical softmax
    wordNgrams=2,
    dim=200,
    thread=2
)
```

**Performance:**
- Test Accuracy: 88.98%
- Precision: 88.98% | Recall: 88.98%

FastText achieves highest accuracy through character n-gram modeling. Subword information helps with rare words and typos common in reviews.

## Comparative Analysis

| Approach | Model | Accuracy | Notes |
|----------|-------|----------|-------|
| BOW + TF-IDF | Naive Bayes | 87.2% | Fast, interpretable baseline |
| BOW + TF-IDF | Logistic Regression | 88.0% | Best traditional approach |
| BOW + TF-IDF | Linear SVM | 86.6% | Reduced features (1K) |
| Word2Vec | Logistic Regression | 88.4% | Dense semantic vectors |
| FastText | Supervised | 88.98% | Best overall, handles OOV |

**Key Observations:**

1. **Marginal Gains:** Embeddings improve accuracy by ~1% over BOW (88.4-89.0% vs 87.2-88.0%). For sentiment analysis, bag-of-words captures sufficient signal.

2. **Semantic Benefits:** Word embeddings provide semantic similarity beyond classification. "good" → "serviceable" relationships enable analogical reasoning unavailable in BOW.

3. **Computational Trade-offs:** BOW with 10K features is sparse but fast. Embeddings are dense (200D) but require pre-training or external models.

4. **Subword Modeling:** FastText's character n-grams handle misspellings and rare words better than Word2Vec. Review data contains informal language where this matters.

5. **Mean Pooling Limitation:** Averaging word vectors loses sequential information. More sophisticated aggregation (weighted averaging, RNN encoding) might improve embedding performance further.

## Prediction Examples

**Negative Review:**
> "The movie was totally boring. The story was dull and the editor did a horrible job at editing this movie."

- BOW (SVM): Negative ✓
- Word2Vec (LR): Negative ✓
- FastText: Negative ✓

All models correctly identify negative sentiment from words like "boring", "dull", "horrible".

**Positive Review:**
> "I loved the way all the Spider-Man movies were pulled into 1 story. I also loved how Spider-Man saved all his villains."

- BOW (SVM): Positive ✓
- Word2Vec (LR): Positive ✓
- FastText: Positive ✓

"Loved" appears twice, providing strong positive signal recognized by all approaches.

## Conclusion

Sentiment classification on movie reviews achieves 87-89% accuracy across BOW and embedding approaches. FastText performs best (88.98%) through subword modeling, while BOW + Logistic Regression provides competitive results (88.0%) with simpler implementation.

The marginal accuracy gain from embeddings (1-2%) must be weighed against increased complexity. For production systems requiring only classification, BOW remains effective. For applications needing semantic similarity (recommendation, search), embeddings justify the additional effort despite modest classification improvement.

Word2Vec and FastText provide interpretable semantic spaces where "good" relates to "serviceable" and "halfdecent". This semantic structure enables downstream tasks beyond classification, making embeddings valuable even when classification accuracy is similar to BOW.

**Full implementation:** [Kaggle Notebook](https://www.kaggle.com/code/mrudhuhas/movie-sentiment-analysis-bow-w2v-fasttext)
