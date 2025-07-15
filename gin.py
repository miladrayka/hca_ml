"""A GIN model."""

from typing import Tuple, Dict, List

import numpy as np
import molgraph
from molgraph import layers, chemistry
import tensorflow as tf
from tensorflow.keras import regularizers


# graph_encoder: A MolecularGraphEncoder that converts SMILES strings into graph objects.
# It uses atom and bond featurizers from molgraph. This encoder is essential
# for creating graph inputs that are usable by GNN layers.

graph_encoder = chemistry.MolecularGraphEncoder(
    atom_encoder=chemistry.Featurizer(
        [
            chemistry.features.Symbol(),
            chemistry.features.Hybridization(),
            chemistry.features.FormalCharge(),
            chemistry.features.TotalNumHs(),
            chemistry.features.TotalValence(),
            chemistry.features.NumRadicalElectrons(),
            chemistry.features.Degree(),
            chemistry.features.ChiralCenter(),
            chemistry.features.Aromatic(),
            chemistry.features.Ring(),
            chemistry.features.Hetero(),
            chemistry.features.HydrogenDonor(),
            chemistry.features.HydrogenAcceptor(),
            chemistry.features.CIPCode(),
            chemistry.features.RingSize(),
            chemistry.features.CrippenLogPContribution(),
            chemistry.features.CrippenMolarRefractivityContribution(),
            chemistry.features.TPSAContribution(),
            chemistry.features.LabuteASAContribution(),
            chemistry.features.GasteigerCharge(),
        ]
    ),
    bond_encoder=chemistry.Featurizer(
        [
            chemistry.features.BondType(),
            chemistry.features.Conjugated(),
            chemistry.features.Rotatable(),
            chemistry.features.Ring(),
            chemistry.features.Stereo(),
        ]
    ),
    self_loops=False,
)


def encode_data(
    graph_encoder: chemistry.MolecularGraphEncoder,
    smiles: List[str],
    labels: List[float],
) -> Tuple:
    """_summary_

    Parameters
    ----------
    graph_encoder : chemistry.MolecularGraphEncoder
        Encodes SMILES strings into molecular graphs and prepares label arrays.
    smiles : List[str]
        List of SMILES.
    labels : List[float]
        List of labels.

    Returns
    -------
    Tuple
        Tuple of graphs and labels.
    """

    mol_graphs = graph_encoder(smiles)
    labels = np.array(labels).astype(np.float32)
    return mol_graphs, labels


def gin_model(
    input_shape: Tuple[int, ...], gnn_hp: Dict, dnn_hp: Dict, training_hp: Dict
) -> tf.keras.Model:
    """Builds a GIN-based graph neural network model for binary classification.

    Parameters
    ----------
    input_shape : Tuple[int, ...]
        Shape of the input data.
    gnn_hp : Dict
        Dictionary of all GIN hyperparameters.
    dnn_hp : Dict
        Dictionary of all DNN hyperparameters.
    training_hp : Dict
        Dictionary of all related hyperparameteres for training.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.

    Raises
    ------
    ValueError
        If an unsupported optimizer is provided.
    """
    model = tf.keras.Sequential()
    model.add(layers.GNNInput(input_shape))
    for i in range(gnn_hp["gnn_num_layers"]):
        model.add(
            molgraph.layers.GINConv(
                units=gnn_hp["gnn_units_list"][i],
                dropout=gnn_hp["gnn_dropout"],
                activation="relu",
                kernel_regularizer=regularizers.l2(training_hp["weight_decay"]),
            )
        )
    model.add(molgraph.layers.Readout(mode="sum"))
    for i in range(dnn_hp["dnn_num_layers"]):
        model.add(
            tf.keras.layers.Dense(
                dnn_hp["dnn_units_list"][i],
                kernel_regularizer=regularizers.l2(training_hp["weight_decay"]),
            )
        )
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(dnn_hp["dnn_dropout"]))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    if training_hp["optimizer_name"] == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=training_hp["learning_rate"])
    elif training_hp["optimizer_name"] == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=training_hp["learning_rate"]
        )
    elif training_hp["optimizer_name"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=training_hp["learning_rate"])
    else:
        raise ValueError(f"Invalid optimizer: {training_hp['learning_rate']}")

    #model.compile(loss="binary_crossentropy", optimizer=optimizer)

    return model
