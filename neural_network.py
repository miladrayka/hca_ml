"""A feedforward neural network model."""

import tensorflow as tf
from tensorflow.keras import regularizers
from typing import Tuple, List


def ffnn_model(
    input_shape: Tuple[int, ...],
    num_layers: int,
    num_neurons_list: List[int],
    activation: str,
    dropout_rate: float,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> tf.keras.Model:
    """
    Creates a feedforward neural network model.

    Parameters
    ----------
    input_shape : Tuple[int, ...]
        Shape of the input data.
    num_layers : int
        Number of hidden layers in the model.
    num_neurons_list : List[int]
        List of integers representing the number of neurons in each hidden layer.
    activation : str
        Activation function to use. Supported values: 'relu', 'leaky_relu', 'elu'.
    dropout_rate : float
        Dropout rate for regularization.
    optimizer_name : str
        Optimizer to use. Supported values: 'sgd', 'rmsprop', 'adam'.
    learning_rate : float
        Learning rate for the optimizer.
    weight_decay : float
        L2 regularization factor.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.

    Raises
    ------
    ValueError
        If an unsupported activation function or optimizer is provided.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))

    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(num_neurons_list[i], kernel_regularizer=regularizers.l2(weight_decay)))
        if activation == 'relu':
            model.add(tf.keras.layers.ReLU())
        elif activation == 'leaky_relu':
            model.add(tf.keras.layers.LeakyReLU(negative_slope=0.01))
        elif activation == 'elu':
            model.add(tf.keras.layers.ELU(alpha=0.1))
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  

    if optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_name}")

    #model.compile(loss='binary_crossentropy',
    #              optimizer=optimizer)

    return model