from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras import regularizers

def build_cnn(embedding_input_dim, embedding_input_length, 
              output_dim, output_activation, objective_function,
              evaluation_metrics, embedding_vector_size=300, 
              convolution_filters=128, convolution_kernel_size=5, 
              hidden_units=300, dropout_rate=0.1,l2=1e-5, optimizer="adam"):
    """Convolutional Neural Network for Text Classification.

    This function builds a CNN model for text classification [1,2]_ 
    on top of Keras.

        [1] Kim, Yoon. "Convolutional neural networks for sentence 
            classification." arXiv preprint arXiv:1408.5882 (2014).
        [2] Zhang, Ye, and Byron Wallace. "A sensitivity analysis of 
            (and practitioners' guide to) convolutional neural networks 
            for sentence classification." arXiv preprint arXiv:1510.03820 
            (2015).

    Args:
        embedding_input_dim (int): Size of the vocabulary including the OOV 
            (Out of Vocabulary) term.
        embedding_input_length (int): Maximum allowed sentence length. Usually 
            set as the length of the largest training sentence.
        output_dim (int): Number of classes for multiclass and multilabel 
            problems or 1 for binary problems.
        output_activation (str): The probability prediction function. Choose 
            'softmax' for multiclass problemas or 'sigmoid' for binary and 
            multilabel problems.
        objective_function (str): The loss function to be minimized. Choose 
            'binary_crossentropy' for binary problems and 
            'categorical_crossentropy' for multiclass and multilabel problems.
        evaluation_metrics (list of str): List of metrics to evaluate model 
            during the training process and to validate it after each epoch.
        embedding_vector_size (int, optional): Dimension of the word embedding 
            vectors. Defaults to 300.
        convolution_filters (int, optional): Number of filters in the 
            convolutional layer. Defaults to 128.
        convolution_kernel_size (int, optional): Size of the word window in
            the convolution, similar to the window size in the Word2Vec
            model. Defaults to 5.
        hidden_units (int, optional): Number of hidden units in the 
            interpretation layer. It is the size of the document
            embedding vectors. Defaults to 300.
        dropout_rate (float, optional): Dropout rate applied to the 
            interpretation layer. Defaults to 0.1.
        l2 (float, optional): Regularization strength to penalize the
            l2 norm of the interpretation layer. Defaults to 1e-5.
        optimizer (str, optional): Optimization algorithm to minimize loss and 
            tune network weights and biases. Defaults to "adam".
    """

    ## Build model
    model = Sequential()
    # 1. Embedding layer to learn word representations
    model.add(Embedding(input_dim=embedding_input_dim,
                        output_dim=embedding_vector_size,
                        input_length=embedding_input_length))
    model.add(Dropout(dropout_rate))
    # 2. Convolutional layer with max pooling to combine words
    model.add(Conv1D(filters=convolution_filters,
                     kernel_size=convolution_kernel_size,
                     strides=1, padding="valid", activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(dropout_rate))
    # 3. Fully connected hidden layer to interpret
    model.add(Dense(units=hidden_units, activation='relu',
                    kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    # 4. Output layer to predict class membership probabilities
    model.add(Dense(units=output_dim, activation=output_activation))

    ## Compile
    model.compile(loss=objective_function, optimizer=optimizer,
                  metrics=evaluation_metrics)

    return(model)