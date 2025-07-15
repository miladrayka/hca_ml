"""Optimize hyperparameters."""

import os
import random
import json
import pickle
from functools import partial
from typing import Dict, Tuple

import optuna
import numpy as np
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

import feature_generation
from utils import calculate_metrics
from neural_network import ffnn_model
import gin


def config_file(params: Dict, filename: str) -> None:
    """Save hyperparameters to a .json file.

    Parameters
    ----------
    params : dict
        Hyperparameters of the algorithm.
    filename : str
        Name of the JSON file.
    """
    with open(filename, "w") as file:
        json.dump(params, file)


def imbalance(fv_dict: Dict) -> Tuple[np.array, np.array]:
    """Balancing data.

    Parameters
    ----------
    fv_dict : Dict
        Feature vector dictionary.

    Returns
    -------
    Tuple[np.array, np.array]
        Oversampled train set.
    """
    X_train = fv_dict["train_fv"]
    y_train = fv_dict["train_label"]

    ros = RandomOverSampler()
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled


def objective(
    trial: object,
    alg: str,
    X_train_list: np.array,
    y_train_list: np.array,
    X_val_list: np.array,
    y_val_list: np.array,
) -> float:
    """Objective function of Optuna package.

    Parameters
    ----------
    trial : object
        Required for Optuna.
    alg : str
        Name of algorithm (logistic, svm, randomforest, xgboost, ffneuralnetwork).
    X_train_list : np.array
        Feature vector train sets.
    y_train_list : np.array
        Label of train sets.
    X_val_list : np.array
        Feature vector of validation sets.
    y_val_list : np.array
        Label of validation sets.

    Returns
    -------
    float
        ROC AUC score of the model.

    Raises
    ------
    ValueError
        If an invalid algorithm is provided.
    """
    auc_list = []
    try:
        for X_train, y_train, X_val, y_val in zip(
            X_train_list, y_train_list, X_val_list, y_val_list
        ):
            if alg == "logistic":
                C = trial.suggest_categorical(
                    "C", [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
                )
                model = LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=10000,
                    n_jobs=-1,
                )

            elif alg == "svm":
                C = trial.suggest_float("C", 1e-3, 1e3, log=True)
                kernel = trial.suggest_categorical(
                    "kernel", ["linear", "poly", "rbf", "sigmoid"]
                )
                gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
                coef0 = trial.suggest_float("coef0", 0, 1, step=None)

                model = SVC(
                    C=C,
                    kernel=kernel,
                    gamma=gamma,
                    coef0=coef0,
                    probability=True,
                    random_state=42,
                )

            elif alg == "randomforest":
                n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
                max_depth = trial.suggest_int("max_depth", 1, 32, step=1)
                min_samples_split = trial.suggest_int(
                    "min_samples_split", 2, 20, step=1
                )
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20, step=1)
                max_features = trial.suggest_categorical(
                    "max_features", [None, "sqrt", "log2"]
                )

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    bootstrap=True,
                    n_jobs=-1,
                    random_state=42,
                )

            elif alg == "xgboost":
                n_estimators = trial.suggest_int("n_estimators", 100, 20000, step=100)
                max_depth = trial.suggest_int("max_depth", 1, 10, step=1)
                learning_rate = trial.suggest_float(
                    "learning_rate", 1e-3, 0.1, log=True
                )
                subsample = trial.suggest_uniform("subsample", 0.05, 1.0)
                colsample_bytree = trial.suggest_uniform("colsample_bytree", 0.05, 1.0)
                min_child_weight = trial.suggest_int("min_child_weight", 1, 20, step=1)

                model = XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_weight=min_child_weight,
                    device="cpu",
                    n_jobs=-1,
                    tree_method="hist",
                    random_state=42,
                )

            elif alg == "ffneuralnetwork":
                num_layers = trial.suggest_int("num_layers", 1, 5)
                num_neurons_list = [
                    trial.suggest_int(f"num_neurons_{i}", 100, 1000, step=100)
                    for i in range(num_layers)
                ]
                activation = trial.suggest_categorical(
                    "activation", ["relu", "leaky_relu", "elu"]
                )
                dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5, step=0.1)
                optimizer_name = trial.suggest_categorical(
                    "optimizer_name", ["sgd", "rmsprop", "adam"]
                )
                learning_rate = trial.suggest_float(
                    "learning_rate", 1e-5, 1e-1, log=True
                )
                weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
                batch_size = trial.suggest_categorical(
                    "batch_size", [32, 64, 128, 256, 512]
                )

                model = ffnn_model(
                    input_shape=(X_train.shape[1],),
                    num_layers=num_layers,
                    num_neurons_list=num_neurons_list,
                    activation=activation,
                    dropout_rate=dropout_rate,
                    optimizer_name=optimizer_name,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                )

                model.compile(
                    loss="binary_crossentropy",
                    optimizer=optimizer_name,
                )

                early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience=20,
                    restore_best_weights=True,
                    min_delta=1e-4,
                )
                model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=200,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=0,
                )

                y_prob = model.predict(X_val).flatten()
                y_pred = (y_prob > 0.5).astype(int)

            elif alg == "gin":

                gnn_num_layers = trial.suggest_int("gnn_num_layers", 2, 5)
                gnn_units_list = [
                    trial.suggest_int(f"gnn_num_neurons_{i}", 64, 256, step=32)
                    for i in range(gnn_num_layers)
                ]
                gnn_dropout = trial.suggest_float("gnn_dropout", 0.2, 0.5, step=0.1)

                dnn_num_layers = trial.suggest_int("dnn_num_layers", 1, 3)
                dnn_units_list = [
                    trial.suggest_int(f"dnn_num_neurons_{i}", 100, 1000, step=100)
                    for i in range(dnn_num_layers)
                ]
                dnn_activation = trial.suggest_categorical("dnn_activation", ["relu"])
                dnn_dropout = trial.suggest_float("dnn_dropout", 0.2, 0.5, step=0.1)

                learning_rate = trial.suggest_float(
                    "learning_rate", 1e-5, 1e-1, log=True
                )
                weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
                batch_size = trial.suggest_categorical(
                    "batch_size", [32, 64, 128, 256, 512]
                )
                optimizer_name = trial.suggest_categorical(
                    "optimizer_name", ["sgd", "rmsprop", "adam"]
                )

                GNN_KWARGS = {
                    "gnn_num_layers": gnn_num_layers,
                    "gnn_units_list": gnn_units_list,
                    "gnn_dropout": gnn_dropout,
                }

                DNN_KWARGS = {
                    "dnn_num_layers": dnn_num_layers,
                    "dnn_units_list": dnn_units_list,
                    "dnn_dropout": dnn_dropout,
                }

                TRAINING_KWARGS = {
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "optimizer_name": optimizer_name,
                }

                model = gin.gin_model(
                    X_train.spec,
                    gnn_hp=GNN_KWARGS,
                    dnn_hp=DNN_KWARGS,
                    training_hp=TRAINING_KWARGS,
                )

                model.compile(
                    loss="binary_crossentropy",
                    optimizer=optimizer_name,
                )

                early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience=20,  # Fixed patience value
                    restore_best_weights=True,
                    min_delta=1e-4,
                )
                model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=200,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=0,
                )

                y_prob = model.predict(X_val).flatten()
                y_pred = (y_prob > 0.5).astype(int)

            else:
                raise ValueError(f"Invalid algorithm: {alg}")

            if alg not in ["ffneuralnetwork", "gin"]:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)[:, 1]

            metrics = calculate_metrics(y_val, y_pred, y_prob, balance=True)
            auc_list.append(metrics["ROC AUC"])

    except Exception as e:
        raise optuna.TrialPruned()
        # print(f"Error: {e}")
        # raise

    return np.mean(auc_list)

def hp_optimize(file_paths: str, alg: str, fv_name: str) -> Tuple[float, Dict]:
    """Optimize hyperparameters by Optuna.

    Parameters
    ----------
    file_paths : str
        Paths to .pkl of inner folds for an outer fold.
    alg : str
        Name of algorithm (logistic, svm, randomforest, xgboost, ffneuralnetwork).
    fv_name : str
        Feature vector dictionary.

    Returns
    -------
    Tuple[float, Dict]
        Return ROC AUC of best model and best hyperparameters.
    """
    imputer = SimpleImputer(strategy="mean", missing_values=np.nan, add_indicator=False)

    EXTREME_VALUE_THRESHOLD = 1e10

    X_train_list = []
    y_train_list = []
    X_val_list = []
    y_val_list = []

    for file_path in file_paths:
        fv = feature_generation.FeatureGeneration(file_path)

        if fv_name == "rdkit":
            fv_des = fv.descriptors()
            X_train, y_train = imbalance(fv_des)
            X_val, y_val = fv_des["val_fv"], fv_des["val_label"]

            X_val = np.where(np.abs(X_val) > EXTREME_VALUE_THRESHOLD, np.nan, X_val)
            imputer.fit(X_train)
            X_val = imputer.transform(X_val)

        elif fv_name == "maccs":
            fv_maccs = fv.maccs()
            X_train, y_train = imbalance(fv_maccs)
            X_val, y_val = fv_maccs["val_fv"], fv_maccs["val_label"]

            X_val = np.where(np.abs(X_val) > EXTREME_VALUE_THRESHOLD, np.nan, X_val)
            imputer.fit(X_train)
            X_val = imputer.transform(X_val)

        elif fv_name == "ecfp":
            fv_ecfp = fv.ecfp()
            X_train, y_train = imbalance(fv_ecfp)
            X_val, y_val = fv_ecfp["val_fv"], fv_ecfp["val_label"]

        elif fv_name == "chemberta":
            fv_chemberta = fv.chemberta()
            X_train, y_train = imbalance(fv_chemberta)
            X_val, y_val = fv_chemberta["val_fv"], fv_chemberta["val_label"]

        elif fv_name == "gin":
            with open(file_path, "rb") as f:
                data_split = pickle.load(f)

            fv_gin = {
                "train_fv": data_split["train_smiles"],
                "train_label": data_split["train_label"],
            }
            ros = RandomOverSampler()
            X_train_resampled, y_train_resampled = ros.fit_resample(
                fv_gin["train_fv"].reshape(-1, 1), fv_gin["train_label"].reshape(-1, 1)
            )
            X_train, y_train = gin.encode_data(
                gin.graph_encoder,
                X_train_resampled.reshape(
                    -1,
                ),
                y_train_resampled.reshape(
                    -1,
                ),
            )
            X_val, y_val = gin.encode_data(
                gin.graph_encoder,
                data_split["val_smiles"],
                data_split["val_label"],
            )

        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_val_list.append(X_val)
        y_val_list.append(y_val)

    objective_partial = partial(
        objective,
        alg=alg,
        X_train_list=X_train_list,
        y_train_list=y_train_list,
        X_val_list=X_val_list,
        y_val_list=y_val_list,
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_partial, n_trials=20, n_jobs=-1)

    return study.best_value, study.best_params
