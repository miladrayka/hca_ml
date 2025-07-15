import pickle
from typing import Dict, List, Tuple

import exmol
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from rdkit import Chem
from rdkit.Chem import AllChem


def predict_with_conformal(
    smiles: str, model_info: Dict[str, Tuple[str, str]], epsilon: float = 0.3
) -> Dict[str, List[int]]:
    """
    Predicts class membership for a molecule using conformal prediction with SVM models.

    Parameters
    ----------
    smiles : str
        Molecule in SMILES format.
    model_info : dict of str to tuple (str, str)
        Mapping of model labels to (model pickle file path, calibration CSV file path).
    epsilon : float, optional
        Threshold for class inclusion in prediction set (default is 0.3).

    Returns
    -------
    dict of str to list of int
        Dictionary mapping model labels to prediction sets (class indices 0 or 1).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    ecfp_fv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    feature_array = np.array(ecfp_fv).reshape(1, -1)

    results = {}

    for label, (model_file, calibration_csv) in model_info.items():
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        cal_scores_df = pd.read_csv(calibration_csv)
        probs = model.predict_proba(feature_array)[0]
        prediction_set = []

        for class_idx, class_name in enumerate(["inactive", "active"]):
            conformity_scores = cal_scores_df[class_name].to_numpy()
            score_test = probs[class_idx]
            p_value = (np.sum(conformity_scores <= score_test) + 1) / (
                len(conformity_scores) + 1
            )
            if p_value > epsilon:
                prediction_set.append(class_idx)

        results[label] = prediction_set

    return results


def model(smiles: str, svm) -> int:
    """
    Predict the binary class of a molecule using an SVM model.

    Parameters
    ----------
    smiles : str
        SMILES string representing the molecule.
    svm : object
        Preloaded SVM model with a `predict` method.

    Returns
    -------
    int
        Predicted class label (0 or 1).
    """
    mol = Chem.MolFromSmiles(smiles)
    ecfp_fv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    y_predict = svm.predict(np.array(ecfp_fv).reshape(1, -1))
    return int(y_predict[0])


def counterfactual_explain(samples, name: str):
    """
    Generate and save counterfactual explanations for molecule samples.

    Parameters
    ----------
    samples : list
        Molecular samples to explain, typically output from `exmol.sample_space`.
    name : str
        Identifier prefix for the saved SVG file.

    Returns
    -------
    None
        Saves an SVG file named '{name}_counterfactual_samples.svg' with counterfactual plots.
    """
    sns.set_context("notebook")
    sns.set_style("dark")

    font_manager.findfont("Helvetica")
    plt.rc("font", family="Helvetica")
    plt.rc("font", serif="Helvetica", size=22)

    fkw = {
        "figsize": (8, 6),
        "dpi": 300,
        "facecolor": "white",
        "edgecolor": "white",
    }

    cfs = exmol.cf_explain(samples, nmols=3)
    exmol.plot_cf(cfs, figure_kwargs=fkw, mol_size=(350, 300), nrows=2, mol_fontsize=8)
    plt.tight_layout()

    svg = exmol.insert_svg(cfs)
    with open(f"{name}_counterfactual_samples.svg", "w") as f:
        f.write(svg)