## References:
# [1] Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
# [2] Zhang, Ye, and Byron Wallace. "A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification." arXiv preprint arXiv:1510.03820 (2015).
# [3] [1D CNN for text classification](https://keras.io/examples/imdb_cnn/).

from pandas import read_csv, DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import sequence
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model
import settings
import logging

## Set parameters
vocab_size = 32768
batch_size = 128
embedding_dims = 64
kernel_size = [3, 5, 7, 9] # Multi-channel CNN for n-grams
filters = 128
hidden_dims = [256, 128]
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
tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(data_train.text)
x_train = tokenizer.texts_to_sequences(data_train.text)
x_test = tokenizer.texts_to_sequences(data_test.text)

## Pad sequences
logging.info("Transforming tokens into sequences...")
max_input_size = len(max(x_train, key = len))
X_train = sequence.pad_sequences(x_train, maxlen = max_input_size)
X_test = sequence.pad_sequences(x_test, maxlen = max_input_size)
print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)
X_train_multi = []
X_test_multi = []
for i in range(len(kernel_size)):
    X_train_multi.append(X_train)
    X_test_multi.append(X_test)

## Build model
# 1. Convolutional channels from word embeddings
inputs = []
channels = []
for i in range(len(kernel_size)):
    inputs.append(Input(shape = (max_input_size, )))
    x = Embedding(vocab_size, embedding_dims)(inputs[i])
    x = Dropout(dropout_prob)(x)
    x = Conv1D(filters, kernel_size[i], activation = 'relu')(x)
    x = Dropout(dropout_prob)(x)
    x = MaxPooling1D()(x)
    channels.append(Flatten()(x))
# 2. Fully connected layers to interpret
x = concatenate(channels)
for hidden_size in hidden_dims:
    x = Dense(hidden_size, activation = 'relu')(x)
    x = Dropout(dropout_prob)(x)
# 3. Softmax output layer
outputs = Dense(len(le.classes_), activation = 'softmax')(x)
model = Model(inputs = inputs, outputs = outputs)

## Compile
model.compile(
    loss      = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics   = ['accuracy']
)
model.summary()
plot_model(model, show_shapes = True, to_file = 'output/multi_cnn_model.png')

## Train network
logging.info("Training network...")
model.fit(
    x               = X_train_multi, 
    y               = Y_train,
    batch_size      = batch_size,
    epochs          = epochs,
    validation_data = (X_test_multi, Y_test)
)
model.save("output/multi_cnn_model")

## Predict test data
logging.info("Predicting test set...")
y_prob = model.predict(X_test_multi)
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
col_names = []
labels = le.classes_
for i in range(len(labels)): 
    col_names.append("prob_{}".format(labels[i]))
data_pred = DataFrame(
    data    = y_prob,
    index   = range(y_prob.shape[0]),
    columns = col_names
)
data_pred["target"] = le.inverse_transform(y_test)
data_pred["pred"] = le.inverse_transform(y_pred)
data_pred.to_csv("output/multi_cnn_prediction.csv")
