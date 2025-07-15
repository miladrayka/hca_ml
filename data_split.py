"""10-fold cross-validation dataset to train, val, and test sets based on
random or scaffold approach."""

import pickle

import numpy as np
import pandas as pd
import deepchem as dc


def categorize_type(file_path: str) -> pd.DataFrame:
    """Categorizes 'pK' column in a csv file based on thresholds.

    Parameters
    ----------
    file_path : str
        The path to the csv file.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the categorized 'pK' column.
    """

    df = pd.read_csv(file_path)

    # Create a new column 'status' based on 'pK' values
    df["status"] = np.nan

    df.loc[(df["has_sulfonamide"]) & (df["pK"] > (9 - np.log10(20))), "status"] = 1
    df.loc[(df["has_sulfonamide"]) & (df["pK"] < 7), "status"] = 0

    return df


file_paths = [
    "../Data/O43570_ChEMBL_data_canonical_smiles_filtered.csv",
    "../Data/P00918_ChEMBL_data_canonical_smiles_filtered.csv",
    "../Data/Q16790_ChEMBL_data_canonical_smiles_filtered.csv",
]

# Calculate number of active and inactive for each UniProt

for file_path in file_paths:
    name = file_path.split("/")[-1].split(".")[0]
    df = categorize_type(file_path)
    df = df.dropna(axis=0)
    df.to_csv(f"../Data/{name}_status.csv", index=False)
    active = df[df.loc[:, "status"] == 1.0].shape[0]
    inactive = df[df.loc[:, "status"] == 0.0].shape[0]
    print(
        f"UniProt {name.split('_')[0]}: \n Active: {active} \n Inactive: {inactive} \n Total: {active + inactive}"
    )


file_paths = [
    "../Data/O43570_ChEMBL_data_canonical_smiles_filtered_status.csv",
    "../Data/P00918_ChEMBL_data_canonical_smiles_filtered_status.csv",
    "../Data/Q16790_ChEMBL_data_canonical_smiles_filtered_status.csv",
]

# 5*5 Fold Nested-Cross-Validation (Random Split)

file_paths = [
    "../Data/O43570_ChEMBL_data_canonical_smiles_filtered_status.csv",
    "../Data/P00918_ChEMBL_data_canonical_smiles_filtered_status.csv",
    "../Data/Q16790_ChEMBL_data_canonical_smiles_filtered_status.csv",
]

for file_path in file_paths:

    name = file_path.split("/")[-1].split("_")[0]

    df = pd.read_csv(file_path)
    x = df.loc[:, "standardize_smiles"].to_numpy()
    y = df.loc[:, "status"].values

    dataset = dc.data.DiskDataset.from_numpy(X=x, y=y, w=np.zeros(len(x)), ids=x)

    random_splitter = dc.splits.RandomSplitter()
    outer_folds = random_splitter.k_fold_split(dataset, k=5, seed=42)
    i = 0
    for outer_train_fold, outer_test_fold in outer_folds:
        x_train_outer, y_train_outer = outer_train_fold.X, outer_train_fold.y
        x_test_outer, y_test_outer = outer_test_fold.X, outer_test_fold.y
        outer_train_test_split_dict = {
            "train_smiles": x_train_outer,
            "train_label": y_train_outer.reshape(-1),
            "test_smiles": x_test_outer,
            "test_label": y_test_outer.reshape(-1),
        }

        with open(
            f"../Data/Data_Splits/Random/{name}_random_outer_{i}.pkl", "wb"
        ) as file:
            pickle.dump(outer_train_test_split_dict, file)
        inner_folds = random_splitter.k_fold_split(outer_train_fold, k=5, seed=42)
        j = 0
        for inner_train_fold, inner_val_fold in inner_folds:
            x_train_inner, y_train_inner = inner_train_fold.X, inner_train_fold.y
            x_val_inner, y_val_inner = inner_val_fold.X, inner_val_fold.y
            inner_train_val_split_dict = {
                "train_smiles": x_train_inner,
                "train_label": y_train_inner.reshape(-1),
                "val_smiles": x_val_inner,
                "val_label": y_val_inner.reshape(-1),
            }

            with open(
                f"../Data/Data_Splits/Random/{name}_random_outer_{i}_inner_{j}.pkl",
                "wb",
            ) as file:
                pickle.dump(inner_train_val_split_dict, file)
                j += 1
        i += 1

# 5*5 Fold Nested-Cross-Validation (Scaffold Split)

for file_path in file_paths:

    name = file_path.split("/")[-1].split("_")[0]

    df = pd.read_csv(file_path)
    x = df.loc[:, "standardize_smiles"].to_numpy()
    y = df.loc[:, "status"].values

    dataset = dc.data.DiskDataset.from_numpy(X=x, y=y, w=np.zeros(len(x)), ids=x)

    scaffold_splitter = dc.splits.ScaffoldSplitter()
    outer_folds = scaffold_splitter.k_fold_split(dataset, k=5, seed=42)
    i = 0
    for outer_train_fold, outer_test_fold in outer_folds:
        x_train_outer, y_train_outer = outer_train_fold.X, outer_train_fold.y
        x_test_outer, y_test_outer = outer_test_fold.X, outer_test_fold.y
        outer_train_test_split_dict = {
            "train_smiles": x_train_outer,
            "train_label": y_train_outer.reshape(-1),
            "test_smiles": x_test_outer,
            "test_label": y_test_outer.reshape(-1),
        }

        with open(
            f"../Data/Data_Splits/Scaffold/{name}_scaffold_outer_{i}.pkl", "wb"
        ) as file:
            pickle.dump(outer_train_test_split_dict, file)
        inner_folds = scaffold_splitter.k_fold_split(outer_train_fold, k=5, seed=42)
        j = 0
        for inner_train_fold, inner_val_fold in inner_folds:
            x_train_inner, y_train_inner = inner_train_fold.X, inner_train_fold.y
            x_val_inner, y_val_inner = inner_val_fold.X, inner_val_fold.y
            inner_train_val_split_dict = {
                "train_smiles": x_train_inner,
                "train_label": y_train_inner.reshape(-1),
                "val_smiles": x_val_inner,
                "val_label": y_val_inner.reshape(-1),
            }

            with open(
                f"../Data/Data_Splits/Scaffold/{name}_scaffold_outer_{i}_inner_{j}.pkl",
                "wb",
            ) as file:
                pickle.dump(inner_train_val_split_dict, file)
                j += 1
        i += 1
