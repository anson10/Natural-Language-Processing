# 🔍 TF-IDF: Term Frequency–Inverse Document Frequency in NLP

TF-IDF (Term Frequency–Inverse Document Frequency) is a **statistical weighting scheme** used to evaluate how important a word is to a document within a collection (corpus). It is fundamental in traditional NLP tasks such as document retrieval, ranking, and clustering.

---

## 📌 Table of Contents

1. [Motivation](#-motivation)
2. [The TF-IDF Formula](#-the-tf-idf-formula)
3. [Variants and Normalizations](#-variants-and-normalizations)
4. [Worked Example](#-worked-example)
5. [Edge Cases & Smoothing](#-edge-cases--smoothing)
6. [Python Implementation](#-python-implementation)
7. [Comparison with Other Techniques](#-comparison-with-other-techniques)
8. [Best Practices](#-best-practices)
9. [References](#-references)

---

## 🎯 Motivation

In Bag-of-Words (BoW), frequent terms dominate regardless of informativeness. Words like _“the”_, _“is”_, _“and”_ appear everywhere, yet they say little about document uniqueness.

TF-IDF mitigates this by:
- **Down-weighting common terms across documents**
- **Up-weighting rare terms within the corpus**

---

## 🧮 The TF-IDF Formula

TF-IDF is defined as:

$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$

### 1. **Term Frequency (TF)**

$$
\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}} = \frac{\text{raw count of } t}{\text{total terms in } d}
$$

Alternative (sublinear scaling):

$$
\text{TF}_{\log}(t, d) = 1 + \log(f_{t,d}), \quad \text{if } f_{t,d} > 0
$$

### 2. **Inverse Document Frequency (IDF)**

Given $N$ documents and $df_t$ (documents containing term $t$):

$$
\text{IDF}(t, D) = \log\left( \frac{1 + N}{1 + df_t} \right) + 1
$$

> This is called **IDF smoothing** — avoids division by 0 and zero log.

---

## 🧪 Variants and Normalizations

TF-IDF vectors are typically **L2 normalized**:

$$
\vec{v}_d = \frac{\vec{v}_d}{\|\vec{v}_d\|_2} = \frac{\vec{v}_d}{\sqrt{\sum_i v_{d_i}^2}}
$$

Why?
- Prevents longer documents from dominating.
- Makes TF-IDF comparable across documents.

Other variants:
- **Binary TF**: $TF(t, d) = 1$ if term appears, else $0$
- **Augmented TF**: $TF(t, d) = 0.5 + 0.5 \cdot \frac{f_{t,d}}{\max_{t'} f_{t',d}}$
- **BM25 weighting**: A probabilistic alternative to TF-IDF used in ranking.

---

## 🧾 Worked Example

### Corpus:

```

Doc1: "the cat sat on the mat"
Doc2: "the dog sat on the log"
Doc3: "dogs and cats are great pets"

````

### Step 1: Raw Term Frequency

| Term  | Doc1 | Doc2 | Doc3 |
|-------|------|------|------|
| cat   | 1    | 0    | 0    |
| dog   | 0    | 1    | 0    |
| sat   | 1    | 1    | 0    |
| the   | 2    | 2    | 0    |

$TF(\text{cat}, \text{Doc1}) = \frac{1}{6}$  
$TF(\text{sat}, \text{Doc2}) = \frac{1}{6}$

---

### Step 2: Document Frequencies

| Term | df |
|------|----|
| cat  | 1  |
| dog  | 1  |
| sat  | 2  |
| the  | 2  |

$N = 3$

$IDF(\text{cat}) = \log\left(\frac{1 + 3}{1 + 1}\right) + 1 = \log(2) + 1 \approx 1.6931$

---

### Step 3: TF-IDF

$$
\text{TF-IDF}(\text{cat}, \text{Doc1}) = \frac{1}{6} \cdot 1.6931 \approx 0.2822
$$

---

## ⚠️ Edge Cases & Smoothing

### Case: Term appears in all documents

If $df_t = N$, then:

$$
\text{IDF}(t) = \log\left( \frac{1 + N}{1 + N} \right) + 1 = \log(1) + 1 = 1
$$

Still non-zero due to smoothing, but has minimal impact.

### Case: Term appears in 0 documents

Never occurs if using a pre-tokenized corpus. But to avoid log(0):

- Always use: $1 + df_t$
- Additive smoothing ensures numerical stability

---

## 🐍 Python Implementation

### Basic Example with `scikit-learn`:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "dogs and cats are great pets"
]

tfidf = TfidfVectorizer(norm='l2', smooth_idf=True, use_idf=True)
X = tfidf.fit_transform(docs)
df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())
print(df.round(3))
````

---

## 🆚 Comparison with Other Techniques

| Technique       | Context-Aware | Uses Frequency | Handles Semantics | Sparse | Use Cases              |
| --------------- | ------------- | -------------- | ----------------- | ------ | ---------------------- |
| Bag of Words    | ❌             | ✅              | ❌                 | ✅      | Baselines, ML models   |
| TF-IDF          | ❌             | ✅              | ❌                 | ✅      | IR, document filtering |
| Word2Vec        | ✅             | ❌              | ✅                 | ❌      | Semantics, similarity  |
| BERT embeddings | ✅             | ❌              | ✅                 | ❌      | Deep NLP, transfer     |

---

## 💡 Best Practices

* **Always lowercase and normalize** (remove punctuation, stop words)
* **Use n-grams** (e.g. bi-grams) if phrases are important
* **Normalize vectors** for clustering and similarity
* **Use feature selection** to reduce dimensionality
* **Combine with ML models**: Naive Bayes, SVM, Logistic Regression

---



