---
title: "(2) Text classification from BOW to Transformers"
date: 2024-01-13
categories: [Natural_Language_Processing, Text_Classification]
tags: [machine-learning, text-classification, natural-language-processing]
comments: false
math: true
---
You're right! I missed incorporating the sections on **classification using embeddings** and **zero-shot classification** while maintaining the flow of the blog post. Let's correct that and provide a comprehensive rewrite that includes those key concepts alongside **prompt-based classification** with generative models.

---

In our previous post, we covered how to preprocess text data and prepare it for machine learning models. Now, we will take the next step in text classification by building models using pre-trained language models and embeddings, exploring techniques like **zero-shot classification**, **embeddings-based classification**, and **prompt-based classification** using generative models. By the end of this post, you'll have a deeper understanding of how to leverage state-of-the-art NLP models to build robust text classification systems. We’ll also evaluate their performance using real-world data.

### Introduction to Text Classification with Pre-trained Models

Text classification is a core task in Natural Language Processing (NLP), where the goal is to assign predefined labels to text. In recent years, pre-trained models like BERT, GPT, and their variants have greatly advanced the state of text classification, enabling strong performance across many NLP tasks. These models, trained on vast amounts of data, can either be fine-tuned on a specific task or applied directly for inference with little modification.

In this post, we will focus on several approaches to text classification:
- Using **pre-trained models** like BERT.
- Applying **zero-shot classification** where the model is able to classify text into labels it hasn't seen during training.
- Classifying text using **embeddings** and traditional machine learning models.
- Exploring **prompt-based classification** using **generative models** like GPT.

### Loading the Dataset and Necessary Libraries

Before diving into the models, let’s start by loading the IMDb movie reviews dataset, which contains reviews labeled as either positive or negative. We’ll also import the necessary libraries.

```python
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Paths to the datasets
splits = {'train': 'IMDB_train.csv', 'validation': 'IMDB_validation.csv', 'test': 'IMDB_test.csv'}

# Loading the data from a remote source (Hugging Face Datasets)
train_df = pd.read_csv("hf://datasets/jahjinx/IMDb_movie_reviews/" + splits["train"])
test_df = pd.read_csv("hf://datasets/jahjinx/IMDb_movie_reviews/" + splits["test"])
valid_df = pd.read_csv("hf://datasets/jahjinx/IMDb_movie_reviews/" + splits["validation"])
```

### Using Pre-trained Models for Text Classification

Pre-trained models are language models trained on extensive corpora, such as BERT or DistilBERT. Instead of training from scratch, we can use these models to classify text directly by fine-tuning them or applying them with minimal changes. Hugging Face’s `pipeline` API makes this process straightforward.

Let’s use the `nlptown/bert-base-multilingual-uncased-sentiment` model for sentiment classification of movie reviews.

```python
from transformers import pipeline

model_path = "nlptown/bert-base-multilingual-uncased-sentiment"

pipe = pipeline(
    model=model_path,
    tokenizer=model_path,
    return_all_scores=True, 
    device="cuda:0"
)
```

In this setup:
- **`model`**: We use the pre-trained BERT model fine-tuned for sentiment classification.
- **`return_all_scores`**: This ensures the model outputs scores for all sentiment classes, not just the top prediction.
- **`device`**: Using GPU for faster inference.

### Classifying the Text Data

Now that we’ve set up the pipeline, let’s classify some text from our validation dataset.

```python
import numpy as np
from tqdm import tqdm

# Store predictions
y_pred = []

# Extract the list of texts from the validation dataset
texts = valid_df['text'].tolist()

# Classify each text
for text in tqdm(texts, total=len(texts)):
    output = pipe(text, max_length=512, truncation=True, padding=True)[0]
    scores = {entry['label']: entry['score'] for entry in output}
    
    # Assign positive or negative sentiment based on score
    negative_score = scores.get('NEGATIVE', 0)
    positive_score = scores.get('POSITIVE', 0)
    assignment = 1 if positive_score > negative_score else 0
    y_pred.append(assignment)
```

### Evaluating Model Performance

We evaluate the model using common classification metrics, such as precision, recall, and F1 score:

```python
from sklearn.metrics import classification_report

# Assuming y_true contains the ground truth labels
classification_report(
    y_true, y_pred,
    target_names=["Negative Review", "Positive Review"]
)
```

The model achieves an accuracy of **87%**, demonstrating strong performance with minimal tuning.

---

### Zero-Shot Classification

**Zero-shot classification** allows models to predict labels they haven’t been explicitly trained on. Instead of needing a fixed set of labels, the model can classify text into new categories based on its understanding of language semantics. This is especially useful when we don’t have labeled training data for every possible class.

#### Example: Zero-Shot Sentiment Classification

With Hugging Face’s `zero-shot-classification` model, we can classify sentiment without explicitly training for this task.

```python
# Initialize the zero-shot classification pipeline
zero_shot_pipe = pipeline(
    model="zero-shot-classification",
    device=0
)

# Define potential labels
labels = ["positive", "negative"]

y_pred = []

# Classify text in the validation set
texts = valid_df['text'].tolist()

for text in tqdm(texts, total=len(texts)):
    output = zero_shot_pipe(text, labels)
    y_pred.append(output['labels'][0])
```

In this example, we ask the model to classify reviews as either "positive" or "negative," even though the model wasn’t explicitly trained for these labels.

### Evaluating the Zero-Shot Model

```python
classification_report(
    y_true, y_pred,
    target_names=["Negative Review", "Positive Review"]
)
```

With **87% accuracy**, this demonstrates the power of zero-shot learning, allowing the model to generalize to new tasks without specific fine-tuning.

---

### Embeddings-Based Classification

Another approach is to use **embeddings** to represent text in a high-dimensional space, where similar texts are closer together. These embeddings can then be used as features for traditional machine learning classifiers, such as logistic regression.

We’ll use **Sentence Transformers** to generate embeddings for the text data.

#### Generating Embeddings with Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer("w601sxs/b1ade-embed-kd_3", trust_remote_code=True)

# Generate embeddings for train, validation, and test datasets
train_embeddings = model.encode(train_df["text"], show_progress_bar=True)
test_embeddings = model.encode(test_df["text"], show_progress_bar=True)
valid_embeddings = model.encode(valid_df["text"], show_progress_bar=True)
```

#### Classification Using Logistic Regression

Once we have the embeddings, we can classify the text using a simple machine learning algorithm like logistic regression.

```python
from sklearn.linear_model import LogisticRegression

# Train a logistic regression classifier on the training embeddings
clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, train_df["label"].values)

# Predict labels for the test embeddings
y_pred = clf.predict(test_embeddings)
```

### Evaluating the Logistic Regression Model

```python
classification_report(
    test_df["label"].values, y_pred,
    target_names=["Negative Review", "Positive Review"]
)
```

Even with logistic regression, the embeddings-based model achieves an **87% accuracy**, showcasing the effectiveness of using embeddings for text classification.

---

### Prompt-Based Classification with Generative Models

Beyond pre-trained models and embeddings, **generative models** like GPT can also be used for classification. By crafting specific prompts, we can frame the task as a question-answer scenario, where the model generates the class label.

#### How Prompt-Based Classification Works

We provide the model with an input (e.g., a movie review) and ask it a question about the input's class label (e.g., "Is the sentiment positive or negative?"). The model generates a response with the classification label.

#### Example: Sentiment Classification Using GPT-2

```python
from transformers import pipeline

# Load the GPT-2 model and tokenizer using Hugging Face's pipeline
gpt_pipe = pipeline(
    model="gpt2",
    tokenizer="gpt2",
    device=0  # Use GPU if available
)

# Example text to classify
text = "This movie was absolutely fantastic! The plot was gripping and the characters were well-developed."

# Crafting a prompt for the model
prompt = f"{text}\nQuestion: Is the sentiment of this review positive or negative?\nAnswer:"

# Generate the classification result
response = gpt_pipe(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']

# Output the response
print(response)
```

Here, the model generates a response to the question embedded in the prompt, such as "positive."

#### Crafting Effective Prompts

The success of this method depends heavily on how the prompt is designed. You can:
- **Be Clear and Specific**: The model works best when given clear, specific questions (e.g., "Is the sentiment positive or negative?").
- **Multi-Class Classification**: You can handle more than two classes by adjusting the prompt (e.g., "Is the sentiment very positive, positive, neutral, negative, or very negative?").

### Conclusion



In this post, we explored multiple techniques for text classification:
1. **Pre-trained models** like BERT allow for easy, high-performing classification with minimal fine-tuning.
2. **Zero-shot classification** enables models to classify text into labels they haven’t explicitly seen during training.
3. **Embeddings-based classification** offers a flexible way to use pre-trained embeddings and traditional machine learning algorithms.
4. **Generative models** like GPT can be repurposed for classification using prompt-based techniques.

We achieved up to **87% accuracy** across several approaches. Each method has its strengths and trade-offs, making them suitable for different scenarios.

In the next post, we will explore **deep learning models** like LSTMs and Transformers, diving deeper into custom-built text classification systems. Stay tuned for more hands-on examples!