import pickle

import exmol
import streamlit as st

from backend import counterfactual_explain, predict_with_conformal, model


st.set_page_config(
    page_title="CAInsight",
    page_icon=r"title_logo.png",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("")
st.header("CAInsight")

st.image(r"Logo.png")

st.sidebar.header("Developer")
st.sidebar.write(
    """[GitHub](https://github.com/miladrayka/hca_ml),
    Developed by *[Milad Rayka](https://scholar.google.com/citations?user=NxF2f0cAAAAJ&hl=en)*."""
)
st.sidebar.divider()
st.sidebar.header("Citation")
st.sidebar.write(
    """**Reference**:
    Paper is *review.*"""
)

st.write(
    """**CAInsight** is an interpretable and uncertainty-aware machine learning software designed to predict the activity and selectivity of human carbonic anhydrase (hCA) isoforms.
      Specifically, we focus on predicting the activity of three isoforms: hCA II, hCA IX, and hCA XII. """
)

with st.expander("**Further Information**"):
    st.info(
        """The primary model relies on a Support Vector Machine (SVM) in conjunction with an Extended Connectivity Fingerprint (ECFP). Each hCA isoform has its own SVM-ECFP binary classifier that returns labels indicating whether they are active or inactive. 
        We enhance our models with [conformal prediction](https://pubs.acs.org/doi/abs/10.1021/ci5001168) (CP), which quantifies the uncertainty in our predictions. In this context, CP can return an active label, an inactive label, a combination of both labels, or an empty set, depending on a specified epsilon value.
        Lastly, we employ counterfactual explainability (see [exmol](https://github.com/ur-whitelab/exmol)) to enhance the interpretability of our model."""
    )

try:
    smiles = st.text_input(
        "**Enter the SMILES**:",
        placeholder="e.g., NS(=O)(=O)c1ccc(NC(=O)Nc2ccc(F)cc2)cc1",
        help="Provide the SMILES for your compund (e.g., NS(=O)(=O)c1ccc(NC(=O)Nc2ccc(F)cc2)cc1).",
    )
except:
    pass

epsilon = st.slider(
    "**Confidence Threshold**:",
    min_value=0.1,
    max_value=0.3,
    step=0.01,
    help="See the definition in the paper.",
)
st.write("**Push the Run button**:")

run = st.button("Run")

if run:
    model_info = {
        "CA2": ("./Data/svm_P00918.pkl", "./Data/svm_ecfp_CA2_prob.csv"),
        "CA9": ("./Data/svm_Q16790.pkl", "./Data/svm_ecfp_CA9_prob.csv"),
        "CA12": ("./Data/svm_O43570.pkl", "./Data/svm_ecfp_CA12_prob.csv"),
    }

    predictions = predict_with_conformal(smiles, model_info, epsilon=epsilon)
    print(predictions)

    isoform_models = {
        "CA2": "./Data/svm_P00918.pkl",
        "CA9": "./Data/svm_Q16790.pkl",
        "CA12": "./Data/svm_O43570.pkl",
    }

    for isoform, model_path in isoform_models.items():
        with open(model_path, "rb") as f:
            svm_model = pickle.load(f)

        model_fn = lambda smiles: model(smiles, svm_model)

        samples = exmol.sample_space(smiles, model_fn, batched=False)
        counterfactual_explain(samples, isoform)

    st.info("**CA2 Explainability**:")
    if predictions["CA2"] == [1]:
        st.info("Conformal Prediction: Active")
    elif predictions["CA2"] == [0]:
        st.info("Conformal Prediction: Inactive")
    elif predictions["CA2"] == [0, 1]:
        st.info("Conformal Prediction: Active or Inactive (Undecidable)")
    else:
        st.info("Conformal Prediction: Empty Set")
    st.image("CA2_counterfactual_samples.svg")
    st.info("**CA9 Explainability**:")
    if predictions["CA9"] == [1]:
        st.info("Conformal Prediction: Active")
    elif predictions["CA9"] == [0]:
        st.info("Conformal Prediction: Inactive")
    elif predictions["CA9"] == [0, 1]:
        st.info("Conformal Prediction: Active or Inactive (Undecidable)")
    else:
        st.info("Conformal Prediction: Empty Set")
    st.image("CA9_counterfactual_samples.svg")
    st.info("**CA12 Explainability**:")
    if predictions["CA12"] == [1]:
        st.info("Conformal Prediction: Active")
    elif predictions["CA12"] == [0]:
        st.info("Conformal Prediction: Inactive")
    elif predictions["CA12"] == [0, 1]:
        st.info("Conformal Prediction: Active or Inactive (Undecidable)")
    else:
        st.info("Conformal Prediction: Empty Set")
    st.image("CA12_counterfactual_samples.svg")
