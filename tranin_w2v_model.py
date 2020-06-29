# References:
# [1] https://rare-technologies.com/word2vec-tutorial/
# [2] https://radimrehurek.com/gensim/models/keyedvectors.html
# [3] https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

from gensim.models import Word2Vec
from pandas import read_csv
from text_utils import tokenize_text
import logging
import settings

# Import data
logging.info("Importing data..")
data_train = read_csv("data/data_train.csv")
data_test = read_csv("data/data_test.csv")
text = data_train.text.append(data_test.text)

# Tokenize text
logging.info("Tokenizing text..")
sentences = [tokenize_text(text) for text in text.values]
sentences.append(["UNK"])

# Train w2v model
logging.info("Training word2vec..")
model = Word2Vec(sentences, size=128, window=5, min_count=1, max_vocab_size=65536, sample=1e-3, iter=5, workers=4)

# Save word embeddings
logging.info("Persisting word vectors on disk..")
model.wv.save("output/word_vectors.kv")

# Model performance
model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
model.wv.doesnt_match("breakfast cereal dinner lunch".split())
model.wv.doesnt_match("cat dog bee horse apple".split())
model.wv.doesnt_match("yellow blue red stone".split())
model.wv.similarity('woman', 'man')
