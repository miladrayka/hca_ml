"""This module contains the following functions:
Standardization to generate canonical SMILES,
check substructure,
calculate min, max, Q1, mean, Q3 and std of some values,
counting the number of IC50, Ki and Kd in the 'type' column in each csv file,
calculate molecular properties and their statistics, and Rule of Five (Ro5),
and T-SNE.
"""

# The following code is adopted from the below link:
# https://docs.datamol.io/stable/tutorials/Preprocessing.html

from typing import Dict, List, Any

import numpy as np
import pandas as pd
import datamol as dm
from rdkit import Chem
from rdkit.Chem import AllChem
from molcomplib import MolCompass
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
)


def canonical_smiles(smiles: str) -> str:
    """The process of standardization is used to generate canonical SMILES.

    Parameters
    ----------
    smiles : str
        SMILES string of a molecule.

    Returns
    -------
    str
        Generate a standardized SMILES.
    """

    mol = dm.to_mol(smiles, ordered=True)
    mol = dm.fix_mol(mol)
    mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
    mol = dm.standardize_mol(
        mol,
        disconnect_metals=False,
        normalize=True,
        uncharge=False,
        stereo=True,
        reionize=True,
    )

    standardize_smiles = dm.standardize_smiles(dm.to_smiles(mol))

    return standardize_smiles


def calculate_statistics(df: pd.DataFrame) -> Dict:
    """Calculate statistics for a single DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        A Panda's DataFrame of a dataset.

    Returns
    -------
    Dict
        Return a dictionary of several statistics: min, max, mean
        q1, q3, std
    """

    values = df.values.flatten()
    min_val = np.min(values)
    q1 = np.percentile(values, 25)
    median = np.median(values)
    q3 = np.percentile(values, 75)
    max_val = np.max(values)
    mean = np.mean(values)
    std = np.std(values)

    stats_dict = {
        "MIN": min_val,
        "Q1": q1,
        "Median": median,
        "Q3": q3,
        "MAX": max_val,
        "Mean": mean,
        "STD": std,
    }

    return stats_dict


def count_letters(df: pd.DataFrame, types: List) -> Dict:
    """Counts occurrences of 'IC50', 'Ki' and 'Kd' in 'type' column.

    Parameters
    ----------
    df : pd.DataFrame
        A Panda's DataFrame of a dataset.
    types : List
        A list of different binding affinity types, i.e., IC50, Ki, and Kd.

    Returns
    -------
    Dict
        A dictionary of numbers of IC50, Ki, and Kd.
    """

    counts = {}

    for t in types:
        filtered_df = df[(df["type"] == t) & (df["has_sulfonamide"])]
        counts[t] = filtered_df.shape[0]
    return counts


def calculate_molecular_property(file_path: str) -> None:
    """Molecular property calculator.

    Parameters
    ----------
    file_path : str
        Path to a csv file.
    """
    df = pd.read_csv(file_path)
    df = df[df.loc[:, "has_sulfonamide"]]
    mols = df["standardize_smiles"].apply(dm.to_mol)

    p_df = dm.descriptors.batch_compute_many_descriptors(mols)
    p_df = p_df.loc[
        :,
        ["mw", "n_lipinski_hba", "n_lipinski_hbd", "qed", "clogp", "n_rotatable_bonds"],
    ]
    p_df.index = df["molecule_chembl_id"]

    name = "../Data/" + file_path.split("/")[-1][:6] + "_molecular_property.csv"
    p_df.to_csv(name, index=True)


def molecular_property_stats(file_path: str) -> None:
    """Calculate statistics for each molecular property.

    Parameters
    ----------
    file_path : str
        Path to a csv file.
    """
    df = pd.read_csv(file_path)
    num_true = ro5(df)
    print(f"{num_true} of {df.shape[0]} are followed Ro5.")
    stats_dict = {}
    for item in [
        "mw",
        "n_lipinski_hba",
        "n_lipinski_hbd",
        "qed",
        "clogp",
        "n_rotatable_bonds",
    ]:
        stats = calculate_statistics(df.loc[:, item])
        stats_dict[item] = stats
    name = (
        "../Data/" + file_path.split("/")[-1][:6] + "_molecular_property_statistics.csv"
    )
    pd.DataFrame(stats_dict).to_csv(name, index=True).round(3)


def ro5(df: pd.DataFrame) -> int:
    """Calculate number molecules that comply the Rule of Five.

    Parameters
    ----------
    df : pd.DataFrame
        A Pandas a dataframe.

    Returns
    -------
    int
        Number of compatible molecules.
    """
    df["Ro5"] = (
        (df["mw"] <= 500)
        & (df["n_lipinski_hba"] <= 10)
        & (df["n_lipinski_hbd"] <= 5)
        & (df["clogp"] <= 5)
    )

    num_true = df["Ro5"].sum()

    return num_true


def t_sne(file_path: str) -> np.array:
    """Perform T-SNE dimensionality reduction.

    Parameters
    ----------
    file_path : str
        A csv file path.

    Returns
    -------
    np.array
        N * 2 matrix.
    """

    df = pd.read_csv(file_path)
    df = df[df["has_sulfonamide"]]
    smiles_list = df.loc[:, "standardize_smiles"].tolist()
    compass = MolCompass()
    reps = np.vstack([compass(smiles) for smiles in smiles_list])

    return reps


def check_sulfonamide(smiles: str) -> bool:
    """Checks if a SMILES string contains the sulfonamide substructure.

    Parameters
    ----------
    smiles : str
        SMILES string of a molecule.

    Returns
    -------
    bool
        True or False value.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False  # Handle invalid SMILES strings

    # Define the SMARTS pattern for sulfonamides
    sulfonamide_pattern = "[#16X4]([NH2])(=[O])(=[O])"

    # Use AllChem.MolFromSmarts to create a query molecule
    query_mol = AllChem.MolFromSmarts(sulfonamide_pattern)

    # Use Allchem.SubstructMatch to check for a match
    match = mol.HasSubstructMatch(query_mol)

    return match


def calculate_metrics(
    y_true: np.array, y_pred: np.array, y_prob: np.array, balance: True
) -> Dict[str, Any]:
    """Calculate balanced classificaion metrics.

    Parameters
    ----------
    y_true : np.Array
        An array of true labels.
    y_pred : np.Array
        An array of predicted labels.
    y_prob : np.Array
        An array of probability values.
    balance: bool, default True
        Adjust metric in inbalance cases.

    Returns
    -------
    Dict[str, Any]
        A dictionary of calculated metrics.
    """
    if balance:
        weights = np.zeros(len(y_true))
        weights[y_true == 0] = len(y_true) / (2 * sum(y_true == 0))
        weights[y_true == 1] = len(y_true) / (2 * sum(y_true == 1))
    else:
        weights = np.ones(len(y_true))

    metrics = {
        "Precision": precision_score(y_true, y_pred, sample_weight=weights),
        "Recall": recall_score(y_true, y_pred, sample_weight=weights),
        "F1-score": f1_score(y_true, y_pred, sample_weight=weights),
        "Accuracy": accuracy_score(y_true, y_pred, sample_weight=weights),
        "ROC AUC": roc_auc_score(y_true, y_prob, sample_weight=weights),
        "Matthews Correlation": 
            matthews_corrcoef(y_true, y_pred, sample_weight=weights)
        ,
    }
    return metrics
