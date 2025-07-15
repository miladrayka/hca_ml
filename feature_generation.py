"""Generate features based on ECFP, RDKit Descriptors, and MACCS."""

import pickle
from typing import Dict, Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel


class FeatureGeneration:
    """
    A class to generate molecular features for machine learning models.

    Attributes
    ----------
    data_split : dict
        Dictionary containing train, validation, and test splits with
          SMILES and labels.
    rdkit_descriptors : list of str
        List of RDKit molecular descriptors to calculate.
    DescCalc : MolecularDescriptorCalculator
        Calculator for RDKit molecular descriptors.
    """

    def __init__(self, pkl_file: str):
        """
        Initialize the FeatureGeneration class with data splits and
          RDKit descriptors.

        Parameters
        ----------
        pkl_file : str
            Path to the pickle file containing train, validation,
              and test splits.
        """
        with open(pkl_file, "rb") as f:
            self.data_split = pickle.load(f)

        with open("./rdkit_descriptors.txt", "r", encoding="utf-8") as file:
            self.rdkit_descriptors = file.read().strip().split(", ")

        self.desc_cal = MolecularDescriptorCalculator(self.rdkit_descriptors)

    def ecfp(self) -> Dict[str, Any]:
        """
        Generate Extended Connectivity Fingerprint (ECFP) feature vectors.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing ECFP feature vectors and labels for train,
              validation, and test splits.
        """
        ecfp_fv_dict = {}
        for split in ["train", "val", "test"]:
            smiles_list = self.data_split.get(f"{split}_smiles", [])
            label_list = self.data_split.get(f"{split}_label", [])

            ecfp_fvs = np.zeros((len(smiles_list), 2048), dtype=float)

            for i, smiles in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"Invalid SMILES string: {smiles}")
                    continue
                ecfp_fvs[i] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

            ecfp_fv_dict[f"{split}_fv"] = ecfp_fvs
            ecfp_fv_dict[f"{split}_label"] = np.array(label_list)

        return ecfp_fv_dict

    def descriptors(self) -> Dict[str, Any]:
        """
        Generate molecular descriptors using RDKit.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing descriptor feature vectors and labels for
              train, validation, and test splits.
        """
        descriptors_fv_dict = {}
        for split in ["train", "val", "test"]:
            smiles_list = self.data_split.get(f"{split}_smiles", [])
            label_list = self.data_split.get(f"{split}_label", [])

            descriptors_fvs = np.zeros(
                (len(smiles_list), len(self.rdkit_descriptors)), dtype=float
            )

            for i, smiles in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"Invalid SMILES string: {smiles}")
                    continue
                descriptors_fvs[i] = list(self.desc_cal.CalcDescriptors(mol))

            descriptors_fv_dict[f"{split}_fv"] = descriptors_fvs
            descriptors_fv_dict[f"{split}_label"] = np.array(label_list)

        descriptors_fv_dict = self._preprocessing(descriptors_fv_dict)
        descriptors_fv_dict = self._scaling(descriptors_fv_dict)

        return descriptors_fv_dict

    def maccs(self) -> Dict[str, Any]:
        """
        Generate MACCS keys feature vectors.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing MACCS keys feature vectors and labels for
              train, validation, and test splits.
        """
        maccs_fv_dict = {}
        for split in ["train", "val", "test"]:
            smiles_list = self.data_split.get(f"{split}_smiles", [])
            label_list = self.data_split.get(f"{split}_label", [])

            maccs_fvs = np.zeros((len(smiles_list), 167), dtype=float)

            for i, smiles in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"Invalid SMILES string: {smiles}")
                    continue
                maccs_fvs[i] = AllChem.GetMACCSKeysFingerprint(mol)

            maccs_fv_dict[f"{split}_fv"] = maccs_fvs
            maccs_fv_dict[f"{split}_label"] = np.array(label_list)

        maccs_fv_dict = self._preprocessing(maccs_fv_dict)
        maccs_fv_dict = self._scaling(maccs_fv_dict)

        return maccs_fv_dict
    
    def chemberta(self) -> Dict[str, Any]:
        """
        Generate ChemBERTa embedding vectors.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing ChemBERTa embeeding vectors and labels for train,
              validation, and test splits.
        """
        tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        chemberta_model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

        chemberta_fv_dict = {}

        for split in ["train", "val", "test"]:
            smiles_list = self.data_split.get(f"{split}_smiles", [])
            label_list = self.data_split.get(f"{split}_label", [])

            chemberta_fvs = np.zeros((len(smiles_list), 384), dtype=float)

            for i, smiles in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"Invalid SMILES string: {smiles}")
                    continue
                inputs = tokenizer(smiles, padding=True, truncation=True, return_tensors='pt', max_length=128)
                with torch.no_grad():
                    chemberta_fvs[i] = chemberta_model(**inputs).last_hidden_state.mean(dim=1).numpy()

            chemberta_fv_dict[f"{split}_fv"] = chemberta_fvs
            chemberta_fv_dict[f"{split}_label"] = np.array(label_list)

        return chemberta_fv_dict

    def _preprocessing(
        self,
        fv_dict: Dict[str, Any],
        var_threshold: float = 0.01,
        corr_threshold: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Preprocess feature vectors by removing low-variance and highly
          correlated features.

        Parameters
        ----------
        fv_dict : Dict[str, Any]
            Dictionary containing feature vectors for train, validation, and
              test splits.
        var_threshold : float, optional
            Variance threshold for feature selection, by default 0.01.
        corr_threshold : float, optional
            Correlation threshold for feature selection, by default 0.95.

        Returns
        -------
        Dict[str, Any]
            The processed feature dictionary with selected features.
        """
        train_data = fv_dict["train_fv"]
        var = np.var(train_data, axis=0)
        selected_features = var > var_threshold
        train_data = train_data[:, selected_features]

        corr_matrix = np.corrcoef(train_data.T)
        upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        to_drop = [
            i
            for i, j in zip(*np.where(np.abs(corr_matrix) > corr_threshold))
            if upper_tri[i, j]
        ]

        for split in ["train", "val", "test"]:
            data = fv_dict[f"{split}_fv"]
            fv_dict[f"{split}_fv"] = np.delete(
                data[:, selected_features], to_drop, axis=1
            )

        return fv_dict

    def _scaling(self, fv_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scale feature vectors using StandardScaler.

        Parameters
        ----------
        fv_dict : Dict[str, Any]
            Dictionary containing feature vectors for train, validation,
              and test splits.

        Returns
        -------
        Dict[str, Any]
            The scaled feature dictionary.
        """
        scaler = StandardScaler()
        fv_dict["train_fv"] = scaler.fit_transform(fv_dict["train_fv"])
        try:
            fv_dict["val_fv"] = scaler.transform(fv_dict["val_fv"])
        except: 
            pass
        try:
            fv_dict["test_fv"] = scaler.transform(fv_dict["test_fv"])
        except:
            pass

        return fv_dict
