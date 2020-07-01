## Setup

Install `Anaconda` or `Miniconda` and:

```bash
conda create -n text-class python=3.7 ipykernel unidecode nltk numpy pandas scikit-learn tensorflow-gpu keras pydot gensim -y
conda activate text-class
```

Running TensorBoard:

```bash
tensorboard --logdir logs --port 6006
```

## Dataset

[AG's News Topic Classification Dataset](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv).

**Stratified splitting:**

- Training set: 102080 samples (80%)
- Test set: 25520 samples (20%)

## Text Preprocessing

```bash
conda activate text-class
python preporcessing.py
conda deactivate
```

**Steps:**

- Lower case.
- Remove accents.
- Remove punctuation.
- Remove numbers.
- Remove single character words.
- Remove english stop-words.
- Remove multiple spaces.
- Remove trailing and padding spaces.

## Experiments

### Tf-Idf + Linear SVM (Stacked with GLM)

```bash
conda activate text-class
python train_svm_model.py
conda deactivate
```

**Parameters:**

- Terms: n-grams from `1` to `3`.
- Min. document frequency: `5`.
- Max. document frequency: `0.5` (50%).
- Vocabulary size: `113149` (all available terms).
- SVM cost: `C = 0.5`.
- Sample weights: inverse of class proportions (`weights='balanced'`).

**Performance:**

- Training time: `00:02:06`.
- Overall Accuracy: `0.8936`.
- Balanced Accuracy: `0.8936`.
- Micro F1-score: `0.8936`.
- Macro F1-score: `0.8934`.
- Log-loss: `0.3312`.

**References:**

- [Classification of text documents using sparse features](https://scikit-learn.org/0.16/auto_examples/text/document_classification_20newsgroups.html).

### Convolutional Neural Network

![](output/cnn_model.png)

### Recurrent Neural Network

### Multi-channel Convolutional Neural Network

![](output/multi_cnn_model.png)

### Character-level Convolutional Neural Network

**References:**

- [Deep Models for NLP beginners](https://github.com/BrambleXu/nlp-beginner-guide-keras).

### Very Deep Convolutional Neural Network

## TO DO

- Create skig-gram model to generate pre-trained embeddings
- Use pre-trained embeddings
- Create files to the Embedding Projector
- Extract document embeddings from the output of the dense layer
- Apply nearest neighbors on document embeddings
- Apply k-means on document embeddings to find topics
- Configure Tensor Board
- Create tests
- Create linter
- Create CI

