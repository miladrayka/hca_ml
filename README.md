[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Interpretable Machine Learning Unveils Carbonic Anhydrase Inhibition via Conformal and Counterfactual Prediction
All the codes to reproduce the paper.

## Citation
For now, please cite this [preprint version](https://chemrxiv.org/engage/chemrxiv/article-details/68a9751da94eede154350c8a).

## Contact
- Milad Rayka, milad.rayka@yahoo.com

- Masoumeh Shams, masoumehshams.gh@gmail.com

## Install

1- Clone *hca_ml* Github repository.
```
git clone https://github.com/miladrayka/hca_ml.git
```

2- Change directory to *hca_ml* and make a new environment from the `cheminf_env.yaml` file by [Mamba](https://github.com/conda-forge/miniforge) package manager:
```
mamba env create -f cheminf_env.yaml
```

## Usage
To reproduce all results, tables, and figures, uncompress the `Data.tar.xz` and `Results.tar.xz` folders and refer to `workflow.ipynb`.

## CAInsight GUI

<img src="https://github.com/miladrayka/hca_ml/blob/main/Logo.png" alt="drawing" width="600" style="display: block; margin: auto;"/>

**CAInsight** is an interpretable and uncertainty-aware machine learning software designed to predict the activity of human carbonic anhydrase (hCA) isoforms. Specifically, we focus on predicting the activity of three isoforms: hCA II, hCA IX, and hCA XII. 

The primary model relies on a Support Vector Machine (SVM) in conjunction with an Extended Connectivity Fingerprint (ECFP). Each hCA isoform has its own SVM-ECFP binary classifier that returns labels indicating whether they are active or inactive. 
We enhance our models with [conformal prediction](https://pubs.acs.org/doi/abs/10.1021/ci5001168) (CP), which quantifies the uncertainty in our predictions. In this context, CP can return an active label, an inactive label, a combination of both labels, or an empty set, depending on a specified epsilon value. Lastly, we employ counterfactual explainability (see [exmol](https://github.com/ur-whitelab/exmol)) to enhance the interpretability of our model.

To run **CAInsight**, change directory to *hca_ml*, then type the following in the terminal:
```
streamlit run gui.py
```

## Copy Right
Copyright (c) 2025, Milad Rayka, Masoumeh Shams
