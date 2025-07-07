
# 📘 NLP Basics: Bag of Words, Stemming, Lemmatization, N-gram, and One-Hot Encoding

This document provides clear and concise explanations of foundational concepts in Natural Language Processing (NLP), including code snippets in Python for better understanding.

---

## 📌 1. Bag of Words (BoW)

### 🔹 What is it?
Bag of Words is a technique to convert text into a numerical feature vector by:
- Removing grammar and word order.
- Counting the frequency of each word in the document.

### 🔹 Example

Sentences:
- "I love NLP"
- "I love learning NLP"

BoW Vocabulary: `["I", "love", "NLP", "learning"]`

| Sentence               | I | love | NLP | learning |
|------------------------|---|------|-----|----------|
| "I love NLP"           | 1 | 1    | 1   | 0        |
| "I love learning NLP"  | 1 | 1    | 1   | 1        |

### 🔹 Python Example
```python
from sklearn.feature_extraction.text import CountVectorizer

sentences = ["I love NLP", "I love learning NLP"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

print(vectorizer.get_feature_names_out())
print(X.toarray())
````

---

## 📌 2. Stemming

### 🔹 What is it?

Stemming reduces words to their root form (stem), often by chopping off suffixes.

* "working" → "work"
* "better" → "better" (not "good", which would be more accurate, see lemmatization)

### 🔹 Python Example

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["working", "worked", "works", "happily"]
stems = [stemmer.stem(word) for word in words]
print(stems)
```

> 📌 Note: Stemming may produce non-dictionary words (e.g., "happily" → "happili").

---

## 📌 3. Lemmatization

### 🔹 What is it?

Lemmatization also reduces words to their base form but **uses vocabulary and grammar rules**, returning dictionary words.

* "working" → "work"
* "better" → "good"

### 🔹 Python Example

```python
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running", pos="v"))  # 'run'
print(lemmatizer.lemmatize("better", pos="a"))   # 'good'
```

---

## 📌 4. N-grams

### 🔹 What is it?

An **n-gram** is a contiguous sequence of `n` words.

* **Unigram (1-gram):** "I", "love", "NLP"
* **Bigram (2-gram):** "I love", "love NLP"
* **Trigram (3-gram):** "I love NLP"

### 🔹 Why use N-grams?

N-grams help capture word sequences and context missed by BoW.

### 🔹 Python Example

```python
from sklearn.feature_extraction.text import CountVectorizer

text = ["I love NLP"]
vectorizer = CountVectorizer(ngram_range=(1, 2))  # unigrams + bigrams
X = vectorizer.fit_transform(text)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

---

## 📌 5. One-Hot Encoding (for text)

### 🔹 What is it?

One-hot encoding represents each word in the vocabulary as a binary vector where **only one position is 1** and the rest are 0.

Vocabulary: \["I", "love", "NLP"]

* "I"     → \[1, 0, 0]
* "love"  → \[0, 1, 0]
* "NLP"   → \[0, 0, 1]

### 🔹 Python Example

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

words = np.array(["I", "love", "NLP"]).reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(words)

print(encoder.categories_)
print(encoded)
```

> 📌 Note: For sequences (like sentences), we typically use tokenization + indexing before one-hot encoding.

---

## 🧠 Summary Table

| Concept          | Purpose                                  | Output Format         |
| ---------------- | ---------------------------------------- | --------------------- |
| Bag of Words     | Count word frequency                     | Sparse matrix         |
| Stemming         | Rule-based word root extraction          | Word roots (raw)      |
| Lemmatization    | Vocabulary-aware word root extraction    | Dictionary root words |
| N-gram           | Capture sequence of words                | Sequence tokens       |
| One-Hot Encoding | Binary vector for each unique word/token | Binary matrix         |

---

## ✅ Prerequisites

```bash
pip install nltk scikit-learn
```

---

## 📚 References

* NLTK Documentation: [https://www.nltk.org](https://www.nltk.org)
* Scikit-learn Feature Extraction: [https://scikit-learn.org/stable/modules/feature\_extraction.html](https://scikit-learn.org/stable/modules/feature_extraction.html)
* Text Analytics: [https://towardsdatascience.com/beginners-guide-to-text-feature-extraction-using-python-8210e7b2d7bf](https://towardsdatascience.com/beginners-guide-to-text-feature-extraction-using-python-8210e7b2d7bf)
