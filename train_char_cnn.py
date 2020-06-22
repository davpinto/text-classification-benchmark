## References:
# [1] Zhang, Xiang, Junbo Zhao, and Yann LeCun. "Character-level convolutional networks for text classification." Advances in neural information processing systems. 2015.
# [2] Conneau, Alexis, et al. "Very deep convolutional networks for text classification." arXiv preprint arXiv:1606.01781 (2016).

from pandas import read_csv, DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.utils.vis_utils import plot_model
from numpy import array, zeros
import settings
import logging

## Set parameters
max_input_size = 1024
batch_size = 128
conv_layers = [[256, 7, 3],
               [256, 7, 3],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, -1],
               [256, 3, 3]]
fully_connected_layers = [1024, 1024]
dropout_prob = 0.5
epochs = 2

## Import data
logging.info("Importing data...")
data_train = read_csv("data/data_train_raw.csv")
data_test = read_csv("data/data_test_raw.csv")

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
tokenizer = Tokenizer(num_words = None, char_level = True, oov_token = "unk")
tokenizer.fit_on_texts(data_train.text)
x_train = tokenizer.texts_to_sequences(data_train.text)
x_test = tokenizer.texts_to_sequences(data_test.text)

## Pad sequences
logging.info("Transforming tokens into sequences...")
X_train = sequence.pad_sequences(x_train, maxlen = max_input_size, padding = "post")
X_test = sequence.pad_sequences(x_test, maxlen = max_input_size, padding = "post")
print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)

## Compute initial embeddings weights
vocab_size = len(tokenizer.word_index)
embedding_weights = []
embedding_weights.append(zeros(vocab_size))
for char, i in tokenizer.word_index.items():
    onehot = zeros(vocab_size)
    onehot[i - 1] = 1
    embedding_weights.append(onehot)
embedding_weights = array(embedding_weights)

## Build model
# 1. Input
inputs = Input(shape = (max_input_size,), name = 'input', dtype = 'int64')
# 2. Embedding layer
embedding_layer = Embedding(vocab_size + 1,
                            vocab_size,
                            input_length = max_input_size,
                            weights = [embedding_weights])
x = embedding_layer(inputs)
# 3. Conv layers
for filter_num, filter_size, pooling_size in conv_layers:
    x = Conv1D(filter_num, filter_size)(x)
    x = Activation('relu')(x)
    if pooling_size != -1:
        x = MaxPooling1D(pool_size=pooling_size)(x)
x = Flatten()(x)
# 4. Fully connected layers
for dense_size in fully_connected_layers:
    x = Dense(dense_size, activation = 'relu')(x)
    x = Dropout(dropout_prob)(x)
# 5. Output layer
predictions = Dense(len(le.classes_), activation = 'softmax')(x)

## Compile model
model = Model(inputs = inputs, outputs = predictions)
model.compile(
    loss      = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics   = ['accuracy']
)
model.summary()
plot_model(model, show_shapes = True, to_file = 'output/char_cnn_model.png')

## Train network
logging.info("Training network...")
model.fit(
    x               = X_train, 
    y               = Y_train,
    batch_size      = batch_size,
    epochs          = epochs,
    validation_data = (X_test, Y_test)
)
model.save("output/char_cnn_model")

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
data_pred.to_csv("output/char_cnn_prediction.csv")
