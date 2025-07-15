import os
import json
import random
from typing import Literal

import numpy as np
import tensorflow as tf

from train_test import Model
from hp_optimization import hp_optimize, config_file


def fix_random_state():
    seed_value = 42
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.compat.v1.set_random_seed(seed_value)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(), config=session_conf
    )
    tf.compat.v1.keras.backend.set_session(sess)


def undo_random_state():
    os.environ.pop("PYTHONHASHSEED", None)
    random.seed(None)
    np.random.seed(None)
    tf.compat.v1.set_random_seed(None)

    if "sess" in globals():
        sess.close()
    tf.compat.v1.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session())


def run_model_pipeline(
    model_type: Literal[
        "logistic", "svm", "randomforest", "xgboost", "ffneuralnetwork", "gin"
    ],
    split_type: Literal["random", "scaffold"],
) -> None:
    """
    Run a full model training and evaluation pipeline for multiple enzymes and fingerprints.

    Parameters
    ----------
    model_type : {"logistic", "svm", "randomforest", "xgboost", "ffneuralnetwork", "gin"}
        The type of model to use. Note that "gin" can only be used when the fingerprint is also "gin".
    split_type : {"random", "scaffold"}
        The type of data split strategy to use.

    Returns
    -------
    None
        This function performs file I/O and model training but does not return any values.
    """
    dict_name = {
        "rdkit": "RDKIT",
        "maccs": "MACCS",
        "ecfp": "ECFP",
        "chemberta": "CHEMBERTA",
        "gin": "GIN",
    }

    dict_en_name = {"CA2": "P00918", "CA9": "Q16790", "CA12": "O43570"}

    model_label = model_type.upper()

    if model_type == "gin":
        total_combinations = len(["CA2", "CA9", "CA12"]) * 1 * 5
    else:
        total_combinations = (
            len(["CA2", "CA9", "CA12"])
            * len(["rdkit", "maccs", "ecfp", "chemberta"])
            * 5
        )
    current_iteration = 0

    print(f"Starting pipeline for {model_type} model with {split_type} splits")
    print(f"Total combinations to process: {total_combinations}")

    for en in ["CA2", "CA9", "CA12"]:
        print(f"\nProcessing enzyme: {en} ({dict_en_name[en]})")

        if model_type == "gin":
            fingerprints = ["gin"]
        else:
            fingerprints = ["rdkit", "maccs", "ecfp", "chemberta"]

        for fn in fingerprints:
            if model_type == "gin" and fn != "gin":
                raise ValueError(
                    "model_type 'gin' can only be used with fingerprint 'gin'"
                )
            if fn == "gin" and model_type != "gin":
                raise ValueError(
                    "fingerprint 'gin' can only be used with model_type 'gin'"
                )

            print(f"\n  Processing fingerprint: {fn}")

            for i in range(5):
                current_iteration += 1
                print(
                    f"\n    Fold {i+1}/5 (Overall progress: {current_iteration}/{total_combinations} - {current_iteration/total_combinations:.1%})"
                )

                file_paths = [
                    f"../Data/Data_Splits/{split_type.capitalize()}/"
                    f"{dict_en_name[en]}_{split_type}_outer_{i}_inner_{j}.pkl"
                    for j in range(5)
                ]

                fix_random_state()
                print("      Starting hyperparameter optimization...")
                best_value, best_params = hp_optimize(file_paths, model_type, fn)
                print(f"      Completed HPO with best value: {best_value:.4f}")

                config_path = (
                    f"../Results/Performance/{en}/{split_type.capitalize()}/"
                    f"{model_label}_{dict_name[fn]}/Fold_{i}/"
                    f"hp_{model_type}_{fn}_outer_{i}.json"
                )
                config_file(best_params, config_path)
                print("      Saved hyperparameters to config file")

                with open(config_path) as file:
                    config = json.load(file)

                undo_random_state()

                outer_file_path = (
                    f"../Data/Data_Splits/{split_type.capitalize()}/"
                    f"{dict_en_name[en]}_{split_type}_outer_{i}.pkl"
                )
                print("      Starting model training and evaluation...")
                model = Model(outer_file_path)
                model.train_test(
                    model_type,
                    fn,
                    config,
                    5,
                    f"../Results/Performance/{en}/{split_type.capitalize()}/"
                    f"{model_label}_{dict_name[fn]}/Fold_{i}",
                )
                print("      Completed model training and evaluation\n")

    print("\nPipeline completed successfully!")
