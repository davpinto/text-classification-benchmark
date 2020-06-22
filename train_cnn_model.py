## References:
# [1] Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
# [2] Zhang, Ye, and Byron Wallace. "A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification." arXiv preprint arXiv:1510.03820 (2015).

from pandas import read_csv, DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.utils.vis_utils import plot_model
import settings
import logging

## Set parameters
vocab_size = 32768
batch_size = 128
embedding_dims = 64 # size of word vectors
kernel_size = 4     # size of word groups in convolution
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
tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(data_train.text)
x_train = tokenizer.texts_to_sequences(data_train.text)
x_test = tokenizer.texts_to_sequences(data_test.text)

## Pad sequences
logging.info("Transforming tokens into sequences...")
max_input_size = len(max(x_train, key = len)) # Max. document length
X_train = sequence.pad_sequences(x_train, maxlen = max_input_size)
X_test = sequence.pad_sequences(x_test, maxlen = max_input_size)
print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)

## Build model
model = Sequential()
# 1. Embedding layer to learn word representations
model.add(Embedding(
    input_dim    = vocab_size,
    output_dim   = embedding_dims,
    input_length = max_input_size
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
model.add(Dense(hidden_dims, activation = 'relu'))
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
model.fit(
    x               = X_train, 
    y               = Y_train,
    batch_size      = batch_size,
    epochs          = epochs,
    validation_data = (X_test, Y_test)
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
data_pred.to_csv("output/cnn_prediction.csv")
