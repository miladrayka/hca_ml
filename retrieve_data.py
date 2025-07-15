"""Retrieve data from ChEMBL database.
The following code is adopted from this GitHub repository:
https://github.com/volkamerlab/teachopencadd/tree/master/teachopencadd/talktorials/T001_query_chembl"""

import math
import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm.auto import tqdm


def pk_converter(standard_value: float) -> float:
    """Convert nM values of Ki or IC50 by 9-log10 to pK values.

    Parameters
    ----------
    standard_value : float
        IC50 or Ki value.

    Returns
    -------
    float
        Converted value of IC50 or Ki in terms of pK.
    """

    pk = 9 - math.log10(standard_value)
    return pk


def download_data(uniprot_id: str) -> None:
    """Retrieve data from ChEMBL by UniProt ID.

    Parameters
    ----------
    uniprot_id : str
        UniProt ID of a protein.
    """

    targets_api = new_client.target
    compounds_api = new_client.molecule
    bioactivities_api = new_client.activity

    targets = targets_api.get(target_components__accession=uniprot_id).only(
        "target_chembl_id", "organism", "pref_name", "target_type"
    )

    targets = pd.DataFrame.from_records(targets)
    chembl_id = targets.iloc[0, 2]
    print(f"The target ChEMBL ID for {uniprot_id} is {chembl_id}.")

    # Create a dictionary bioactivity data for each type

    bioactivities_data = {
        "IC50": [],
        "Ki": [],
        "Kd": [],
    }

    for i in ["IC50", "Ki", "Kd"]:
        for j in ["=", ">"]:
            bioactivities = bioactivities_api.filter(
                target_chembl_id=chembl_id, type=i, relation=j, assay_type="B"
            ).only(
                "activity_id",
                "assay_chembl_id",
                "assay_description",
                "assay_type",
                "molecule_chembl_id",
                "type",
                "standard_units",
                "relation",
                "standard_value",
                "target_chembl_id",
                "target_organism",
            )

            for item in bioactivities:
                bioactivities_data[i].append(item)

    # Create DataFrames for each bioactivity type
    IC50_df = pd.DataFrame.from_records(bioactivities_data["IC50"])
    Ki_df = pd.DataFrame.from_records(bioactivities_data["Ki"])
    Kd_df = pd.DataFrame.from_records(bioactivities_data["Kd"])

    # Preprocess
    # Drop some columns and convert 'standard_value' to float

    IC50_df.drop(["units", "value"], axis=1, inplace=True)
    IC50_df = IC50_df.astype({"standard_value": "float64"})
    IC50_df.dropna(axis=0, how="any", inplace=True)
    IC50_df = IC50_df[IC50_df["standard_units"] == "nM"]

    Ki_df.drop(["units", "value"], axis=1, inplace=True)
    Ki_df = Ki_df.astype({"standard_value": "float64"})
    Ki_df.dropna(axis=0, how="any", inplace=True)
    Ki_df = Ki_df[Ki_df["standard_units"] == "nM"]

    Kd_df.drop(["units", "value"], axis=1, inplace=True)
    Kd_df = Kd_df.astype({"standard_value": "float64"})
    Kd_df.dropna(axis=0, how="any", inplace=True)
    Kd_df = Kd_df[Kd_df["standard_units"] == "nM"]

    bioactivities_dfs = pd.concat([IC50_df, Ki_df, Kd_df])

    duplicate_molecule_chembl_id = bioactivities_dfs.loc[
        bioactivities_dfs.duplicated(["molecule_chembl_id"]), "molecule_chembl_id"
    ].unique()

    for id in duplicate_molecule_chembl_id:
        duplicate_standard_value = bioactivities_dfs.loc[
            bioactivities_dfs.loc[:, "molecule_chembl_id"] == id, "standard_value"
        ]
        median = duplicate_standard_value.median().item()
        bioactivities_dfs.loc[
            bioactivities_dfs.loc[:, "molecule_chembl_id"] == id, "standard_value"
        ] = median

    bioactivities_dfs.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)
    print(f"DataFrame shape: {bioactivities_dfs.shape}")

    bioactivities_dfs.reset_index(drop=True, inplace=True)

    compounds_provider = compounds_api.filter(
        molecule_chembl_id__in=list(bioactivities_dfs["molecule_chembl_id"])
    ).only("molecule_chembl_id", "molecule_structures")

    compounds = list(tqdm(compounds_provider))

    compounds_df = pd.DataFrame.from_records(
        compounds,
    )
    compounds_df.dropna(axis=0, how="any", inplace=True)
    compounds_df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)

    canonical_smiles = []

    for i, j in compounds_df.iterrows():
        try:
            canonical_smiles.append(j["molecule_structures"]["canonical_smiles"])
        except KeyError:
            canonical_smiles.append(None)

    compounds_df["smiles"] = canonical_smiles
    compounds_df.drop("molecule_structures", axis=1, inplace=True)
    compounds_df.dropna(axis=0, how="any", inplace=True)

    output_df = pd.merge(
        bioactivities_dfs[
            [
                "molecule_chembl_id",
                "relation",
                "standard_value",
                "type",
                "standard_units",
            ]
        ],
        compounds_df,
        on="molecule_chembl_id",
    )
    output_df.reset_index(drop=True, inplace=True)
    print(f"Dataset with {output_df.shape[0]} entries.")

    try:
        output_df["pK"] = output_df.apply(
            lambda x: pk_converter(x.standard_value), axis=1
        )

    except Exception:
        dis_idx = output_df[output_df.loc[:, "standard_value"] == 0].index
        output_df.drop(dis_idx, axis=0, inplace=True).reset_index(
            drop=True, inplace=True
        )
        output_df["pK"] = output_df.apply(
            lambda x: pk_converter(x.standard_value), axis=1
        )

    output_df.to_csv(f"../Data/{uniprot_id}_ChEMBL_data.csv", index=False)
