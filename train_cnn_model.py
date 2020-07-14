## References:
# [1] Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
# [2] Zhang, Ye, and Byron Wallace. "A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification." arXiv preprint arXiv:1510.03820 (2015).

from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from numpy import vstack, arange, append
from keras.utils import np_utils
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from benchmark.models import build_cnn
from datetime import datetime
from joblib import dump
import settings
import logging

## Set parameters
vocab_size = 32768
batch_size = 128
embedding_dims = 64 # size of word vectors
kernel_size = 4     # size of word groups in convolution (like window size in W2V and GloVe)
filters = 128
hidden_dims = 256
dropout_prob = 0.25
epochs = 2

## Import data
logging.info("Importing data...")
data_train = read_csv("data/data_train.csv", index_col=0)
data_test = read_csv("data/data_test.csv", index_col=0)

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

## Build model
model = build_cnn(embedding_input_dim=vocab_size + 1, 
                  embedding_input_length=max_input_size,
                  output_dim=len(le.classes_), output_activation="sigmoid",
                  objective_function="categorical_crossentropy",
                  evaluation_metrics=["accuracy"])
model.summary()
plot_model(model, show_shapes=True, to_file='output/cnn_model.png')

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
model.save("output/cnn")

## Predict test data
logging.info("Predicting test set...")
y_prob = model.predict(X_test)
y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
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
logging.info("Extracting word embeddings...")
words = DataFrame.from_dict(tokenizer.index_word, orient='index', columns=["word"])
words = words[:(vocab_size + 1)]
embeddings = model.layers[0].get_weights()[0]
nrow, ncol = embeddings.shape
col_names = ["embedding_{:02d}".format(i+1) for i in range(ncol)]
embeddings = DataFrame(embeddings, columns = col_names, index = words.index)
embeddings = concat([words, embeddings], axis = 1, sort=False)
embeddings.to_csv("output/cnn_word_embeddings.csv")
embeddings.drop('word', axis=1, inplace=False).to_csv("output/cnn_embedding_vectors.tsv", sep="\t", header=False, index=False)
embeddings.word.to_csv("output/cnn_embedding_metadata.tsv", sep="\t", header=False, index=False)

## Extract document embeddings
# Doc: https://keras.io/getting_started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction
# PS: Extracting all samples at the same time will freeze TensorFlow.
logging.info("Extracting document embeddings...")
model = load_model("output/cnn")
embedding_extractor = Model(
    inputs  = model.input, 
    outputs = model.get_layer("dense").output
)
nrow, ncol = X_test.shape
idx = arange(start=0, stop=nrow, step=batch_size)
if idx.max() < nrow: idx = append(idx, nrow)
embeddings = [embedding_extractor(X_test[idx[i]:idx[i+1]]).numpy() for i in range(len(idx)-1)]
embeddings = vstack(tuple(embeddings))
col_names = ["embedding_{:02d}".format(i+1) for i in range(hidden_dims)]
embeddings = DataFrame(embeddings, columns = col_names, index = data_test.index)
embeddings.to_csv("output/cnn_doc_embeddings_test_data.csv")
