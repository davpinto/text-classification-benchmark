## References:
# [1] Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
# [2] Zhang, Ye, and Byron Wallace. "A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification." arXiv preprint arXiv:1510.03820 (2015).

from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from gensim.models import KeyedVectors
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras import regularizers
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard
from numpy import zeros, vstack
from datetime import datetime
from joblib import dump
import settings
import logging

## Set parameters
vocab_size = 32768
batch_size = 128
embedding_dims = 256 # size of word vectors
kernel_size = 4      # size of word groups in convolution (like window size in W2V and GloVe)
filters = 128
hidden_dims = 256
dropout_prob = 0.25
epochs = 2

## Import data
logging.info("Importing data...")
data_train = read_csv("data/data_train.csv")
data_test = read_csv("data/data_test.csv")

## Encode output
logging.info("Encoding output...")
le = LabelEncoder()
le.fit(data_train.category.unique())
y_train = le.transform(data_train.category)
y_test = le.transform(data_test.category)
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

## Tokenize text
logging.info("Tokenizing text...")
tokenizer = Tokenizer(num_words = vocab_size, oov_token = "UNK")
tokenizer.fit_on_texts(data_train.text)
dump(tokenizer, "output/tokenizer.joblib", compress=1)
x_train = tokenizer.texts_to_sequences(data_train.text)
x_test = tokenizer.texts_to_sequences(data_test.text)

## Pad sequences
logging.info("Transforming tokens into sequences...")
max_input_size = len(max(x_train, key = len)) # Max. document length
X_train = sequence.pad_sequences(x_train, maxlen = max_input_size)
X_test = sequence.pad_sequences(x_test, maxlen = max_input_size)
print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)

## Import word vectors
logging.info("Importing pre-trained word embeddings...")
wv = KeyedVectors.load("output/word_vectors.kv")

## Build model
model = Sequential()
# 1. Embedding layer to learn word representations
wt = wv[list(tokenizer.index_word.values())[1:(vocab_size + 1)]]
wt = vstack([zeros(wt.shape[1]), wt])
model.add(Embedding(
    input_dim    = vocab_size + 1,
    output_dim   = embedding_dims,
    input_length = max_input_size,
    weights      = [wt], 
    trainable    = True
))
model.add(Dropout(dropout_prob))
# 2. Convolutional layer with max pooling to combine words
model.add(Conv1D(
    filters,
    kernel_size,
    strides    = 1,
    padding    = "valid",
    activation = "relu"
))
model.add(Dropout(dropout_prob))
model.add(GlobalMaxPooling1D())
# 3. Fully connected hidden layer to interpret
model.add(Dense(
    units                = hidden_dims, 
    activation           = 'relu',
    kernel_regularizer   = regularizers.l2(1e-5),
    bias_regularizer     = regularizers.l2(1e-5),
    activity_regularizer = regularizers.l2(1e-5)
))
model.add(Dropout(dropout_prob))
# 4. Softmax output layer
model.add(Dense(len(le.classes_), activation = 'softmax'))

## Compile
model.compile(
    loss      = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics   = ['accuracy']
)
model.summary()
plot_model(model, show_shapes = True, to_file = 'output/cnn_model.png')

## Train network
logging.info("Training network...")
logdir = "logs/cnn/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(
    log_dir        = logdir, 
    histogram_freq = 1, 
    batch_size     = batch_size, 
    write_graph    = True, 
    write_grads    = False
)
model.fit(
    x               = X_train, 
    y               = Y_train,
    batch_size      = batch_size,
    epochs          = epochs,
    validation_data = (X_test, Y_test), 
    callbacks       = [tensorboard_callback]
)
model.save("output/cnn_model")

## Predict test data
logging.info("Predicting test set...")
y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=-1)
logging.info("Overall Accuracy: {:.2f}%".format(
    100 * metrics.accuracy_score(y_test, y_pred)
))
logging.info("Balanced Accuracy: {:.2f}%".format(
    100 * metrics.balanced_accuracy_score(y_test, y_pred)
))
logging.info("Micro F1-score: {:.2f}%".format(
    100 * metrics.f1_score(y_test, y_pred, average = "micro")
))
logging.info("Macro F1-score: {:.2f}%".format(
    100 * metrics.f1_score(y_test, y_pred, average = "macro")
))
logging.info("Log-loss: {:.5f}".format(
    metrics.log_loss(y_test, y_prob)
))

## Save predictions
logging.info("Persisting predictions on disk...")
col_names = ["prob_{}".format(label) for label in le.classes_]
data_pred = DataFrame(
    data    = y_prob,
    index   = range(y_prob.shape[0]),
    columns = col_names
)
data_pred["target"] = le.inverse_transform(y_test)
data_pred["pred"] = le.inverse_transform(y_pred)
data_pred.to_csv("output/cnn_prediction.csv")

## Extract word embeddings
words = DataFrame.from_dict(tokenizer.index_word, orient='index', columns=["word"])
words = words[:vocab_size]
embeddings = model.layers[0].get_weights()[0]
col_names = ["embedding_{:02d}".format(i+1) for i in range(embeddings.shape[1])]
embeddings = DataFrame(embeddings, columns = col_names, index = words.index)
embeddings = concat([words, embeddings], axis = 1, sort=False)
embeddings.to_csv("output/cnn_word_embeddings.csv")
embeddings.drop('word', axis=1, inplace=False).to_csv("output/cnn_embedding_vectors.tsv", sep="\t", header=False, index=False)
embeddings.word.to_csv("output/cnn_embedding_metadata.tsv", sep="\t", header=False, index=False)
