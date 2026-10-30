# IMDB Movie Review Sentiment Analysis

This repository contains a Jupyter notebook that builds a binary sentiment classifier for IMDB movie reviews. The notebook demonstrates a complete machine-learning workflow: loading labeled data, cleaning text, turning text into numbers, creating PyTorch datasets, defining a neural network, training it, and measuring test accuracy.

> **Important architecture note:** The notebook is named as an RNN project, but it does not feed a sequence of words into the recurrent layer. It converts each whole review into one 5,000-feature TF-IDF vector, then supplies that vector as a sequence of length one. The model therefore has an RNN layer, but does not learn word order or dependencies between words over time. This README explains the notebook as it is implemented and points out what would be needed for a true sequence-based RNN.

## Contents

- [Repository files](#repository-files)
- [Problem and data](#problem-and-data)
- [Environment and running the notebook](#environment-and-running-the-notebook)
- [Notebook walkthrough](#notebook-walkthrough)
  - [1. Import pandas and read the CSV](#1-import-pandas-and-read-the-csv)
  - [2. Inspect and clean the dataset](#2-inspect-and-clean-the-dataset)
  - [3. Preprocess review text](#3-preprocess-review-text)
  - [4. Encode sentiment labels](#4-encode-sentiment-labels)
  - [5. Convert reviews to TF-IDF vectors](#5-convert-reviews-to-tf-idf-vectors)
  - [6. Split the data](#6-split-the-data)
  - [7. Create PyTorch datasets and loaders](#7-create-pytorch-datasets-and-loaders)
  - [8. Define the RNN classifier](#8-define-the-rnn-classifier)
  - [9. Configure loss and optimizer](#9-configure-loss-and-optimizer)
  - [10. Train the model](#10-train-the-model)
  - [11. Evaluate the model](#11-evaluate-the-model)
- [How to interpret the result](#how-to-interpret-the-result)
- [Limitations and improvement ideas](#limitations-and-improvement-ideas)

## Repository files

| File | Description |
|---|---|
| `RNN_for_sentimentanalysis.ipynb` | Notebook containing the code, output examples, model training, and evaluation. |
| `IMDB Dataset.csv` | Input data read by the notebook. The notebook expects it in the current working directory. |
| `README.md` | Project overview and explanation of the notebook's code and concepts. |

## Problem and data

The task is **binary sentiment classification**. Given a movie review, predict whether its sentiment is positive or negative:

| Sentiment | Encoded target |
|---|---:|
| Negative | 0 |
| Positive | 1 |

The CSV has two columns:

- `review`: the review text.
- `sentiment`: its known `positive` or `negative` label.

The notebook displays 50,000 rows before duplicate removal, with no missing values in the two columns. After removing duplicate rows, the displayed dataset contains 49,582 examples. A row is one labeled example used to teach the model how text relates to sentiment.

## Environment and running the notebook

The notebook imports and uses:

- Python
- pandas
- scikit-learn
- NLTK
- PyTorch
- Jupyter Notebook or JupyterLab

Install the packages into the same Python environment that is selected as the notebook kernel. A basic CPU example is:

```bash
python -m pip install pandas scikit-learn nltk torch notebook
```

For GPU-enabled PyTorch, use the installation command appropriate for the machine's operating system, GPU, and CUDA setup.

To run:

1. Clone or download the repository.
2. Keep `IMDB Dataset.csv` alongside the notebook, or change the notebook's CSV path.
3. Open `RNN_for_sentimentanalysis.ipynb` in Jupyter.
4. Select the Python environment where the dependencies are installed.
5. Run cells from top to bottom.

The NLTK download cell requests `punkt`, `punkt_tab`, and `stopwords`. The first run may need an internet connection to download them.

## Notebook walkthrough

### 1. Import pandas and read the CSV

The notebook begins by importing pandas, a library for working with tabular data:

```python
import pandas as pd
```

`pd` is the conventional short alias used later to call pandas functions. The CSV is loaded into a `DataFrame`:

```python
df = pd.read_csv("IMDB Dataset.csv")
```

`read_csv` parses the comma-separated file and infers its columns. The variable `df` now contains the review text and sentiment label for each row. The path is relative, so it is resolved against the notebook's current working directory.

### 2. Inspect and clean the dataset

The first inspection statements are:

```python
df.shape
df.head()
df.isnull().sum()
```

- `df.shape` returns `(rows, columns)`. Its saved output is `(50000, 2)`.
- `df.head()` displays the first five examples so the column names, label values, and raw review format can be checked.
- `df.isnull().sum()` counts missing values separately for each column. The saved output shows zero missing values for both `review` and `sentiment`.

The notebook then removes duplicate rows:

```python
df.drop_duplicates(inplace=True)
df.shape
```

`drop_duplicates` removes rows whose values duplicate a previous row. `inplace=True` updates `df` instead of returning a separate cleaned copy. The following shape check displays `(49582, 2)`, meaning 418 duplicate rows were removed from the original 50,000.

### 3. Preprocess review text

Text preprocessing makes input more consistent before vocabulary features are built. These operations affect only the `review` column; the sentiment labels are kept for supervised learning.

#### 3.1 Convert to lowercase

```python
df["review"] = df["review"].str.lower()
```

The string accessor `.str` applies string operations to every value in the pandas column. Lowercasing means, for example, `GOOD` and `good` do not become separate vocabulary entries later.

#### 3.2 Import regular expressions and remove URLs

```python
import re

def remove_urls(text):
    text = re.sub(r"http\S+", "", text)
    return text

df["review"] = df["review"].apply(remove_urls)
```

- `re` is Python's regular-expression module.
- `re.sub(pattern, replacement, text)` replaces every matching part of a string.
- `r"http\S+"` matches the characters `http` followed by one or more non-whitespace characters. The `r` makes this a raw string so backslashes in regex patterns are interpreted as regex syntax.
- The replacement is an empty string, so matching URL-like text is removed.
- `Series.apply` calls `remove_urls` on each review and stores the returned text back in the column.

The notebook also has a commented-out tiny `re.sub` example. Since it is commented, it is explanatory only and does not change the dataset.

#### 3.3 Remove punctuation and non-ASCII characters

```python
def remove_punctuations(text):
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text

df["review"] = df["review"].apply(remove_punctuations)
```

Inside the character class, `A-Z` and `a-z` mean English uppercase and lowercase letters, `0-9` means digits, and `\s` means whitespace. The `^` at the start of the character class negates it, so the pattern matches characters that are **not** in those groups. Replacing them with `""` removes punctuation and characters outside the specified ASCII range.

This is a simple cleanup rule, not a language-aware tokenizer. It also removes punctuation that can express emphasis, contractions, or sentence boundaries.

#### 3.4 Remove HTML-like tags

```python
def remove_html(text):
    text = re.sub(r"<.*?>", "", text)
    return text

df["review"] = df["review"].apply(remove_html)
```

The pattern `<.*?>` matches text from `<` to the next `>`; `.*?` is a non-greedy match, so it stops at the nearest closing angle bracket. The goal is to remove markup such as HTML tags.

**Order caveat:** In the notebook, punctuation removal occurs before this HTML-removal function. For source text such as `<br />`, the angle brackets and slash are removed first, leaving letters like `br`; the later `remove_html` function can no longer match that fragment. The saved cleaned examples still show `br br`. A robust order is to remove or parse HTML before stripping punctuation.

#### 3.5 Download NLTK data and import tokenization tools

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

The downloads provide NLTK resources used by the notebook. The next imports are:

```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
```

- `word_tokenize` splits a text string into word-like tokens.
- `stopwords` provides a built-in list of common words for a language.

There is a commented sample demonstrating tokenization. Because it is commented out, it is not run and has no effect on the dataset.

#### 3.6 Remove English stop words

```python
def remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")

    for word in tokens:
        if word in stop_words:
            text = text.replace(word, "")

    return text

df["review"] = df["review"].apply(remove_stopwords)
```

This function tokenizes one review, obtains NLTK's English stop-word list, checks each token, and attempts to remove stop words from the original string.

Stop words are common words such as articles and prepositions. Removing them can reduce vocabulary size, but it is a task-dependent choice. In sentiment classification, words such as `not` can reverse sentiment, so removing negation can damage the signal.

**Implementation caveat:** `text.replace(word, "")` is a substring replacement on the entire review. It does not remove only a complete token. If a stop word appears as part of a larger word, that substring can be removed too. It may also produce extra spaces. A safer approach is to filter the token list by exact membership and then join the remaining tokens.

The notebook's saved output visibly shows words altered by this operation. This can impact the vocabulary and model predictions.

#### 3.7 Stem the remaining tokens

```python
from nltk.stem import PorterStemmer

def stemming(text):
    ps = PorterStemmer()
    stemmed_words = []

    tokens = word_tokenize(text)
    for token in tokens:
        stemmed_token = ps.stem(token)
        stemmed_words.append(stemmed_token)

    return " ".join(stemmed_words)

df["review"] = df["review"].apply(stemming)
```

Stemming maps related word forms toward a common stem. For example, `played` and `playing` may be reduced toward `play`. `PorterStemmer` applies English suffix-stripping rules; a stem is not always a valid standalone dictionary word.

The code tokenizes the review, stems each token, collects the results, and joins them back into a space-separated string. Stemming can combine related forms into shared features, though it can also reduce linguistic detail.

### 4. Encode sentiment labels

The labels are strings, but the neural-network loss expects numeric targets. The notebook uses scikit-learn's `LabelEncoder`:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["sentiment"] = le.fit_transform(df["sentiment"])
y = df["sentiment"]
```

- `LabelEncoder()` creates the encoder.
- `fit_transform` learns the distinct class names and converts each into a numeric class ID.
- `y` stores the resulting target values, one per review.

For the two labels in this dataset, the notebook's displayed values establish the mapping: `negative` is `0`, and `positive` is `1`. A later call to `.values` obtains the underlying NumPy array for constructing a PyTorch tensor.

### 5. Convert reviews to TF-IDF vectors

The notebook imports and configures a text vectorizer:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(max_features=5000)
X = tf.fit_transform(df["review"])
```

Text cannot be passed directly to this neural-network layer, so `TfidfVectorizer` maps it to numbers. It builds a vocabulary and produces one feature per selected term, up to 5,000 features.

**TF-IDF** means term frequency–inverse document frequency:

- Term frequency reflects how often a term appears in a review.
- Inverse document frequency downweights terms that appear in many reviews and gives relatively more weight to terms that are more distinctive.

`fit_transform` both learns the vocabulary and IDF weights and transforms the reviews into a feature matrix. `X` is initially a sparse matrix because most reviews contain only a small fraction of the full vocabulary. Its conceptual shape is:

```text
(number of reviews, 5000)
```

TF-IDF represents which terms are important in a review; it does not preserve the original order of words.

**Evaluation caveat:** This vectorizer is fitted on all reviews before the train/test split. Therefore, its vocabulary and IDF statistics include information from test review text. Test labels are not used in fitting TF-IDF, but best practice is to split first, fit the vectorizer on training reviews only, then call `transform` on test reviews.

### 6. Split the data

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

`train_test_split` keeps paired feature rows and labels aligned while randomly dividing them:

- `test_size=0.2` assigns 20% to testing and 80% to training.
- `random_state=42` makes the partition repeatable.
- `X_train` and `y_train` are used to learn model parameters.
- `X_test` and `y_test` are held back for evaluation.

The saved shapes are:

| Split | Feature shape |
|---|---:|
| Training | `(39665, 5000)` |
| Testing | `(9917, 5000)` |

The code does not pass `stratify=y`; a stratified split is often preferable for classification because it maintains the approximate class proportions in each partition. A fixed split does not by itself make neural-network weight initialization or training fully reproducible.

### 7. Create PyTorch datasets and loaders

The notebook imports PyTorch and its data utilities:

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
```

The sparse TF-IDF matrices are converted to dense arrays:

```python
X_train = X_train.toarray()
X_test = X_test.toarray()
```

`.toarray()` materializes every feature value, including zeros. This is straightforward for a small example but can consume a lot of memory for larger datasets.

The arrays and labels are wrapped as tensors and paired into datasets:

```python
train_set = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train.values).float()
)

test_set = TensorDataset(
    torch.from_numpy(X_test).float(),
    torch.from_numpy(y_test.values).float()
)
```

- `torch.from_numpy` creates a tensor sharing the NumPy array's data where possible.
- `.float()` converts the tensor to 32-bit floating point.
- `TensorDataset` pairs each review feature row with the matching label. Indexing the dataset returns an `(input, target)` pair.
- Targets are floats because binary cross-entropy compares probability values against targets in the range 0 to 1.

The loaders batch examples:

```python
train_loader = DataLoader(train_set, shuffle=True, batch_size=64)
test_loader = DataLoader(test_set, shuffle=True, batch_size=64)
```

Each iteration yields up to 64 examples. Shuffling training data changes its order between epochs, which is commonly used during optimization. Test shuffling is not needed for accuracy because predictions are compared to their corresponding labels within each batch; it does not change the aggregate metric.

### 8. Define the RNN classifier

The notebook imports neural-network layers and an optimizer:

```python
import torch.nn as nn
import torch.optim as optim
```

The model is defined as a PyTorch module:

```python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
```

#### Constructor: `__init__`

- `input_size` is the number of input features at each sequence step. It is later set to 5,000.
- `hidden_size=128` gives the RNN 128 hidden-state values.
- `num_layers=1` creates one recurrent layer.
- `super().__init__()` initializes the `nn.Module` machinery so PyTorch can track parameters and layers.
- `self.rnn` constructs a vanilla recurrent layer. `batch_first=True` means its input layout is `(batch, sequence, features)`.
- `self.fc` is a fully connected linear layer that maps the 128-dimensional final state to one scalar logit.

#### Forward pass: `forward`

For input batch `x`, `x.size(0)` is the batch size. `h0` is initialized to zero with shape:

```text
(number of layers, batch size, hidden size)
= (1, batch size, 128)
```

`self.rnn(x, h0)` returns:

- `out`: the hidden output at every time step, shaped `(batch, sequence length, 128)`.
- The second result: the final hidden state, unused here and assigned to `_`.

`out[:, -1, :]` selects the output from the last sequence step for every batch item, giving `(batch, 128)`. `self.fc(...)` transforms that to `(batch, 1)`. It returns a raw logit; sigmoid is applied later.

#### RNN concept and the sequence-length detail

A vanilla RNN updates a hidden state at each time step. A simplified recurrence is:

```text
h_t = tanh(x_t W_ih^T + h_(t-1) W_hh^T + b)
```

`x_t` is the input at step `t`; `h_(t-1)` is the previous hidden state; the weight matrices and bias are learned during training. In a sequence of words, the hidden state can carry a summary of earlier words into later steps.

This notebook does **not** use one time step per word. During training it calls `Xb.unsqueeze(1)`, so the feature shape changes from:

```text
(batch, 5000) → (batch, 1, 5000)
```

The RNN therefore sees one input step containing all 5,000 TF-IDF features. It has no word-by-word sequence over which to propagate context. It can still learn a nonlinear classification function, but this arrangement does not model word order.

### 9. Configure loss and optimizer

```python
input_size = X_train.shape[1]

model = RNN(input_size)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())
```

- `X_train.shape[1]` is the feature count, 5,000, so it configures the RNN input size.
- `RNN(input_size)` creates the model with the default 128 hidden units and one layer.
- `nn.BCELoss()` calculates binary cross-entropy. It expects probabilities, not raw logits, and targets with values 0 or 1.
- `optim.Adam(model.parameters())` creates the Adam optimization algorithm for all trainable model parameters. The notebook uses Adam's default learning rate and other defaults.

The model's linear output is a logit. The training and evaluation code applies the sigmoid:

```text
probability = sigmoid(logit) = 1 / (1 + exp(-logit))
```

This converts an unrestricted real number into a value between 0 and 1, interpreted as the probability of positive sentiment. PyTorch code can alternatively use raw logits with `nn.BCEWithLogitsLoss()`, which combines sigmoid and binary cross-entropy in a numerically stable operation.

### 10. Train the model

The training code is:

```python
epochs = 10

for epoch in range(epochs):
    model.train()

    for Xb, yb in train_loader:
        optimizer.zero_grad()

        Xb = Xb.unsqueeze(1)
        outputs = model(Xb)
        outputs = torch.sigmoid(outputs.squeeze())

        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

    print(f"epoch = {epoch+1}/{epochs} and loss = {loss.item()}")
```

#### What happens in each part

1. `epochs = 10` sets the number of passes over the training set.
2. The outer loop repeats training for each epoch.
3. `model.train()` sets the module to training mode. This is important for layers with train/evaluation-specific behavior such as dropout or batch normalization; this simple model has neither, but the pattern is standard.
4. `for Xb, yb in train_loader` obtains a batch of feature vectors and matching labels.
5. `optimizer.zero_grad()` clears gradients left from the previous batch. PyTorch accumulates gradients by default.
6. `Xb.unsqueeze(1)` adds a sequence dimension, transforming `(batch, 5000)` into `(batch, 1, 5000)`.
7. `model(Xb)` calls the model's `forward` method and returns a `(batch, 1)` logit tensor.
8. `outputs.squeeze()` removes dimensions of size one; `torch.sigmoid` converts logits to probabilities.
9. `criterion(outputs, yb)` computes the batch loss by comparing probabilities to the target labels.
10. `loss.backward()` computes gradients of the loss with respect to trainable parameters through backpropagation.
11. `optimizer.step()` uses those gradients to update the parameters.
12. The print statement reports the loss value after the inner batch loop.

**Loss reporting caveat:** The printed `loss.item()` is the loss for only the **last batch** of each epoch, not the mean loss over the epoch. It should not be treated as an epoch-wide average learning curve. The saved notebook output shows the last-batch loss from one run. Since the notebook does not explicitly seed PyTorch's model initialization, a new run may produce different values.

The use of `.squeeze()` is fine for the usual batch sizes here, but if a batch ever contains exactly one example it can remove the batch dimension too. More robust code can use `outputs.squeeze(-1)` to remove only the final output dimension.

### 11. Evaluate the model

The evaluation cell is:

```python
model.eval()

with torch.no_grad():
    correct_vals = 0
    tot_vals = 0

    for Xb, yb in test_loader:
        Xb = Xb.unsqueeze(1)

        outputs = model(Xb)
        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()

        tot_vals += yb.size(0)
        correct_vals += (predicted == yb).sum().item()

    print(f"accuracy = {correct_vals/tot_vals*100}")
```

- `model.eval()` switches to evaluation mode.
- `torch.no_grad()` disables gradient tracking, saving memory and computation because the model is not being updated.
- The input batch gets the same sequence dimension as during training.
- `torch.sigmoid` converts logits to probabilities.
- `> 0.5` applies the decision threshold. A value above 0.5 predicts class 1 (positive); otherwise the prediction is class 0 (negative).
- `.float()` makes predicted labels the same numeric type as `yb`.
- `yb.size(0)` is the batch size and is added to the total example count.
- `(predicted == yb)` produces one Boolean per example; `.sum()` counts correct predictions.
- `.item()` converts the scalar tensor count into a normal Python number.
- The final fraction of correct predictions is multiplied by 100 to report a percentage.

The output saved in the notebook reports:

```text
accuracy = 85.15680145205204
```

This is about **85.16% accuracy** for the saved run and test split. Accuracy is the fraction of all examples predicted correctly. It does not show whether errors are false positives or false negatives, so a confusion matrix, precision, recall, and F1 score would give a more complete picture.

## How to interpret the result

The accuracy belongs to the entire pipeline as implemented: its duplicate removal, text-cleaning order, stop-word replacement, stemming, TF-IDF vectorizer, one-step RNN model, random train/test split, and one saved training run. It is not proof that the model learned word-order relationships. The TF-IDF representation supplies weighted term features, and all terms enter the RNN together in one time step.

The vectorizer was also fitted before splitting, so test review text contributed to the vocabulary and IDF calculation. For a strict evaluation, fit preprocessing only on training data and then apply the learned transformation to test data.

## Limitations and improvement ideas

These are recommendations for a future revision; they are not steps currently implemented in the notebook:

1. **Split before fitting TF-IDF.** Split raw review text and labels first, call `fit_transform` on training text, and call `transform` on test text.
2. **Fix preprocessing order.** Remove HTML before punctuation and use token-aware cleanup. Filter stop words by exact token membership rather than `str.replace`.
3. **Preserve sentiment negation.** Carefully decide whether to retain words such as “not,” since they can reverse sentiment.
4. **Use an actual word sequence for a recurrent model.** Tokenize each review into token IDs, map IDs to embeddings, pad or pack batches, and give the RNN multiple time steps. An LSTM or GRU is another common recurrent choice.
5. **Compare a TF-IDF baseline.** Logistic regression or a linear support vector classifier is a useful baseline for bag-of-words/TF-IDF features.
6. **Improve evaluation.** Use a stratified split, track average loss across all training batches, and report precision, recall, F1, and a confusion matrix in addition to accuracy.
7. **Avoid unnecessary dense conversion at scale.** Converting sparse TF-IDF to dense arrays can use substantial memory.
8. **Improve reproducibility.** Seed Python, NumPy, and PyTorch as appropriate and record library versions and hardware.

## End-to-end data flow

```text
CSV review text
    → lowercase
    → remove URL-like strings
    → remove punctuation/non-ASCII characters
    → attempt HTML removal
    → remove English stop words
    → Porter stemming
    → TF-IDF vector (up to 5,000 features)
    → reshape as one-step input (batch, 1, 5,000)
    → RNN (128 hidden units)
    → linear output logit
    → sigmoid probability
    → threshold at 0.5
    → predicted positive/negative sentiment

CSV sentiment label
    → LabelEncoder (negative=0, positive=1)
    → binary-cross-entropy target
    → compare with prediction during training/evaluation
```