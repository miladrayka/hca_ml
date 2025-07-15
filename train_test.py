"""Train, validate, and test different models."""

import os
import pickle
import joblib
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler

from tensorflow.keras.callbacks import EarlyStopping

import feature_generation
from utils import calculate_metrics
from neural_network import ffnn_model
import gin


class Model(object):
    def __init__(self, file_path: str):
        """Initiate a model.

        Parameters
        ----------
        file_path : str
            Path to .pkl of dataset split.
        """
        self.file_path = file_path
        self.imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
        self.EXTREME_VALUE_THRESHOLD = 1e10  # Configurable threshold

        with open(file_path, "rb") as f:
            self.data_split = pickle.load(f)

        fv = feature_generation.FeatureGeneration(file_path)
        self.fv_des = fv.descriptors()
        self.fv_maccs = fv.maccs()
        self.fv_ecfp = fv.ecfp()
        self.fv_chemberta = fv.chemberta()

    def _handle_extreme_values(self, X_train: np.array, X_data: np.array) -> np.array:
        """Handle extreme values and NaN in data using imputer fitted on training data.

        Parameters
        ----------
        X_train : np.array
            Training data to fit the imputer
        X_data : np.array
            Data to transform (val or test)

        Returns
        -------
        np.array
            Transformed data with extreme values and NaN handled
        """
        # Replace extreme values with NaN
        X_data = np.where(np.abs(X_data) > self.EXTREME_VALUE_THRESHOLD, np.nan, X_data)

        # Fit imputer on training data and transform the input data
        self.imputer.fit(X_train)
        return self.imputer.transform(X_data)

    def train_test(
        self, alg: str, fv_name: str, config: Dict, num: int, path_to_save: str
    ) -> None:
        """Train, validate, and test.

        Parameters
        ----------
        alg : str
            Name of algorithm (logistic, svm, randomforest, xgboost, ffneuralnetwork, gin).
        fv_name : str
            Name of feature vector (rdkit, maccs, ecfp, chemberta, gin).
        config : Dict
            Hyperparameters dictionary for the employed algorithm.
        num : int
            Number of repetition of training for different random number.
        path_to_save : str
            Path to save results.
        """
        if fv_name == "rdkit":
            X_train, y_train = self.imbalance(self.fv_des)
            X_val, y_val = self.fv_des["val_fv"], self.fv_des["val_label"]
            X_test, y_test = self.fv_des["test_fv"], self.fv_des["test_label"]

            # Handle extreme values for rdkit
            try:
                X_val = self._handle_extreme_values(X_train, X_val)
            except:
                pass
            X_test = self._handle_extreme_values(X_train, X_test)

        elif fv_name == "maccs":
            X_train, y_train = self.imbalance(self.fv_maccs)
            X_val, y_val = self.fv_maccs["val_fv"], self.fv_maccs["val_label"]
            X_test, y_test = self.fv_maccs["test_fv"], self.fv_maccs["test_label"]

            # Handle extreme values for maccs
            try:
                X_val = self._handle_extreme_values(X_train, X_val)
            except:
                pass
            X_test = self._handle_extreme_values(X_train, X_test)

        elif fv_name == "ecfp":
            X_train, y_train = self.imbalance(self.fv_ecfp)
            X_val, y_val = self.fv_ecfp["val_fv"], self.fv_ecfp["val_label"]
            X_test, y_test = self.fv_ecfp["test_fv"], self.fv_ecfp["test_label"]

        elif fv_name == "chemberta":
            X_train, y_train = self.imbalance(self.fv_chemberta)
            X_val, y_val = self.fv_chemberta["val_fv"], self.fv_chemberta["val_label"]
            X_test, y_test = (
                self.fv_chemberta["test_fv"],
                self.fv_chemberta["test_label"],
            )

        elif fv_name == "gin":
            # fv_gin = {
            #    "train_fv": self.data_split["train_smiles"],
            #    "train_label": self.data_split["train_label"],
            # }
            # ros = RandomOverSampler()
            # X_train_resampled, y_train_resampled = ros.fit_resample(
            #    fv_gin["train_fv"].reshape(-1, 1), fv_gin["train_label"].reshape(-1, 1)
            # )
            # X_train, y_train = gin.encode_data(
            #    gin.graph_encoder,
            #    X_train_resampled.reshape(-1,),
            #    y_train_resampled.reshape(-1,),
            # )
            # try:
            #    X_val, y_val = gin.encode_data(
            #        gin.graph_encoder,
            #        self.data_split["val_smiles"],
            #        self.data_split["val_label"],
            #    )
            # except:
            #    pass
            # X_test, y_test = gin.encode_data(
            #    gin.graph_encoder,
            #    self.data_split["test_smiles"],
            #    self.data_split["test_label"],
            # )

            fv_gin = {
                "train_fv": self.data_split["train_smiles"],
                "train_label": self.data_split["train_label"],
            }

            train_set_size = int(self.data_split["train_smiles"].shape[0])
            validation_ratio = int(self.data_split["train_smiles"].shape[0] * 0.1)
            validation_index = np.random.choice(train_set_size, validation_ratio)

            self.data_split["val_smiles"] = self.data_split["train_smiles"][
                validation_index
            ]
            self.data_split["val_label"] = self.data_split["train_label"][
                validation_index
            ]

            fv_gin["val_fv"] = self.data_split["val_smiles"]
            fv_gin["val_label"] = self.data_split["val_label"]

            self.data_split["train_smiles"] = np.delete(
                self.data_split["train_smiles"], validation_index
            )
            self.data_split["train_label"] = np.delete(
                self.data_split["train_label"], validation_index
            )

            fv_gin["train_fv"] = self.data_split["train_smiles"]
            fv_gin["train_label"] = self.data_split["train_label"]

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
                fv_gin["val_fv"].reshape(
                    -1,
                ),
                fv_gin["val_label"].reshape(
                    -1,
                ),
            )
            X_test, y_test = gin.encode_data(
                gin.graph_encoder,
                self.data_split["test_smiles"],
                self.data_split["test_label"].reshape(
                    -1,
                ),
            )
        else:
            raise ValueError(f"Invalid feature vector name: {fv_name}")

        if alg == "logistic":
            models = [LogisticRegression(**config) for _ in range(num)]
        elif alg == "svm":
            models = [SVC(**config, probability=True) for _ in range(num)]
        elif alg == "randomforest":
            models = [RandomForestClassifier(**config) for _ in range(num)]
        elif alg == "xgboost":
            models = [XGBClassifier(**config) for _ in range(num)]
        elif alg == "ffneuralnetwork":
            models = [
                ffnn_model(
                    input_shape=(X_train.shape[1],),
                    num_layers=config["num_layers"],
                    num_neurons_list=[
                        config[f"num_neurons_{i}"] for i in range(config["num_layers"])
                    ],
                    activation=config["activation"],
                    dropout_rate=config["dropout_rate"],
                    optimizer_name=config["optimizer_name"],
                    learning_rate=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                )
                for _ in range(num)
            ]
        elif alg == "gin":
            GNN_KWARGS = {
                "gnn_num_layers": config["gnn_num_layers"],
                "gnn_units_list": [
                    config[f"gnn_num_neurons_{i}"]
                    for i in range(config["gnn_num_layers"])
                ],
                "gnn_dropout": config["gnn_dropout"],
            }

            DNN_KWARGS = {
                "dnn_num_layers": config["dnn_num_layers"],
                "dnn_units_list": [
                    config[f"dnn_num_neurons_{i}"]
                    for i in range(config["dnn_num_layers"])
                ],
                "dnn_dropout": config["dnn_dropout"],
            }

            TRAINING_KWARGS = {
                "learning_rate": config["learning_rate"],
                "weight_decay": config["weight_decay"],
                "batch_size": config["batch_size"],
                "optimizer_name": config["optimizer_name"],
            }
            models = [
                gin.gin_model(
                    X_train.spec,
                    gnn_hp=GNN_KWARGS,
                    dnn_hp=DNN_KWARGS,
                    training_hp=TRAINING_KWARGS,
                )
                for _ in range(num)
            ]
        else:
            raise ValueError(f"Invalid algorithm: {alg}")

        if alg == "ffneuralnetwork":
            for i, model in enumerate(models):
                model.compile(
                    loss="binary_crossentropy",
                    optimizer=TRAINING_KWARGS["optimizer_name"],
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
                    validation_split=0.2,
                    epochs=200,
                    batch_size=TRAINING_KWARGS["batch_size"],
                    callbacks=[early_stopping],
                    verbose=0,
                )

                model.save(os.path.join(path_to_save, f"{alg}_{fv_name}_{i}.h5"))
            try:
                self._predict_ffnn(
                    models, fv_name, X_val, y_val, "val", num, path_to_save
                )
            except:
                pass
            self._predict_ffnn(
                models, fv_name, X_test, y_test, "test", num, path_to_save
            )
        elif alg == "gin":
            for i, model in enumerate(models):
                model.compile(
                    loss="binary_crossentropy",
                    optimizer=TRAINING_KWARGS["optimizer_name"],
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
                    batch_size=TRAINING_KWARGS["batch_size"],
                    callbacks=[early_stopping],
                    verbose=0,
                )

                model.save(os.path.join(path_to_save, f"gin_{i}.h5"))
            try:
                self._predict_gin(models, X_val, y_val, "val", num, path_to_save)
            except:
                pass
            self._predict_gin(models, X_test, y_test, "test", num, path_to_save)
        else:
            trained_models = [model.fit(X_train, y_train) for model in models]
            [
                joblib.dump(
                    model, os.path.join(path_to_save, f"{alg}_{fv_name}_{i}.pkl")
                )
                for i, model in enumerate(trained_models)
            ]
            try:
                self._predict(
                    alg, trained_models, fv_name, X_val, y_val, "val", num, path_to_save
                )
            except:
                pass
            self._predict(
                alg, trained_models, fv_name, X_test, y_test, "test", num, path_to_save
            )

    def imbalance(self, fv_dict: Dict) -> Tuple[np.array, np.array]:
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

    def _predict(
        self,
        alg: str,
        models: List,
        fv_name: str,
        X: np.array,
        y: np.array,
        set_name: str,
        num: int,
        path_to_save: str,
    ) -> None:
        """Predict using trained models and save metrics.

        Parameters
        ----------
        alg : str
            Name of algorithm (logistic, svm, randomforest, xgboost).
        models : List
            List of trained models.
        fv_name : str
            Name of feature vector (rdkit, maccs, ecfp, chemberta).
        X : np.array
            Feature vector.
        y : np.array
            Label vector.
        set_name : str
            Test or val sets.
        num : int
            Number of repetition of training for different random number.
        path_to_save : str
            Path to save results.
        """
        metrics_dict = {}

        y_prob_list = []
        y_pred_list = []
        for i in range(num):
            y_pred = models[i].predict(X)
            y_prob = models[i].predict_proba(X)[:, 1]
            metrics = calculate_metrics(y, y_pred, y_prob, balance=True)
            metrics_dict[i] = metrics
            y_prob_list.append(y_prob)
            y_pred_list.append(y_pred)

        df_pred = (
            pd.DataFrame(y_pred_list)
            .transpose()
            .rename(columns={i: f"y_pred_{i}" for i in range(num)})
        )
        df_prob = (
            pd.DataFrame(y_prob_list)
            .transpose()
            .rename(columns={i: f"y_prob_{i}" for i in range(num)})
        )
        df_prob[f"y_{set_name}"] = y
        pd.concat([df_pred, df_prob], axis=1).to_csv(
            os.path.join(path_to_save, f"{set_name}_{alg}_{fv_name}.csv")
        )

        pd.DataFrame(metrics_dict).to_csv(
            os.path.join(path_to_save, f"metrics_{set_name}_{alg}_{fv_name}.csv"),
            index=True,
        )

    def _predict_ffnn(
        self,
        models: List,
        fv_name: str,
        X: np.array,
        y: np.array,
        set_name: str,
        num: int,
        path_to_save: str,
    ) -> None:
        """Predict using trained FFNN models and save metrics.

        Parameters
        ----------
        models : List
            List of trained FFNN models.
        fv_name : str
            Name of feature vector (rdkit, maccs, ecfp, chemberta).
        X : np.array
            Feature vector.
        y : np.array
            Label vector.
        set_name : str
            Test or val sets.
        num : int
            Number of repetition of training for different random number.
        path_to_save : str
            Path to save results.
        """
        metrics_dict = {}

        y_prob_list = []
        y_pred_list = []
        for i in range(num):
            y_prob = models[i].predict(X).flatten()
            y_pred = (y_prob > 0.5).astype(int)
            metrics = calculate_metrics(y, y_pred, y_prob, balance=True)
            metrics_dict[i] = metrics
            y_prob_list.append(y_prob)
            y_pred_list.append(y_pred)

        df_pred = (
            pd.DataFrame(y_pred_list)
            .transpose()
            .rename(columns={i: f"y_pred_{i}" for i in range(num)})
        )
        df_prob = (
            pd.DataFrame(y_prob_list)
            .transpose()
            .rename(columns={i: f"y_prob_{i}" for i in range(num)})
        )
        df_prob[f"y_{set_name}"] = y
        pd.concat([df_pred, df_prob], axis=1).to_csv(
            os.path.join(path_to_save, f"{set_name}_ffneuralnetwork_{fv_name}.csv")
        )

        pd.DataFrame(metrics_dict).to_csv(
            os.path.join(
                path_to_save, f"metrics_{set_name}_ffneuralnetwork_{fv_name}.csv"
            ),
            index=True,
        )

    def _predict_gin(
        self,
        models: List,
        X: np.array,
        y: np.array,
        set_name: str,
        num: int,
        path_to_save: str,
    ) -> None:
        """Predict using trained GIN models and save metrics.

        Parameters
        ----------
        models : List
            List of trained GIN models.
        X : np.array
            Feature vector.
        y : np.array
            Label vector.
        set_name : str
            Test or val sets.
        num : int
            Number of repetition of training for different random number.
        path_to_save : str
            Path to save results.
        """
        metrics_dict = {}

        y_prob_list = []
        y_pred_list = []
        for i in range(num):
            y_prob = models[i].predict(X).flatten()
            y_pred = (y_prob > 0.5).astype(int)
            metrics = calculate_metrics(y, y_pred, y_prob, balance=True)
            metrics_dict[i] = metrics
            y_prob_list.append(y_prob)
            y_pred_list.append(y_pred)

        df_pred = (
            pd.DataFrame(y_pred_list)
            .transpose()
            .rename(columns={i: f"y_pred_{i}" for i in range(num)})
        )
        df_prob = (
            pd.DataFrame(y_prob_list)
            .transpose()
            .rename(columns={i: f"y_prob_{i}" for i in range(num)})
        )
        df_prob[f"y_{set_name}"] = y
        pd.concat([df_pred, df_prob], axis=1).to_csv(
            os.path.join(path_to_save, f"{set_name}_gin.csv")
        )

        pd.DataFrame(metrics_dict).to_csv(
            os.path.join(path_to_save, f"metrics_{set_name}_gin.csv"),
            index=True,
        )
