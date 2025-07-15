import os
import pandas as pd
import numpy as np
from scipy.stats import mode
from statsmodels.stats.contingency_tables import mcnemar
import itertools

# Constants
BASE_DIR = "../Results/Performance"
ISOFORM_NAMES = ["CA2", "CA9", "CA12"]
SPLIT_TYPE = "Scaffold"
FOLD_PREFIX = "Fold_"
NUM_FOLDS = 5
PREDICTION_COLUMNS = [f"y_pred_{i}" for i in range(NUM_FOLDS)]
LABEL_COLUMN = "y_test"


def parse_model_name(model_dir_name):
    """Split folder name into algorithm and feature type."""
    if model_dir_name.upper() == "GIN_GIN":
        return "GIN", "GIN"

    for sep in ["-", "_"]:
        if sep in model_dir_name:
            parts = model_dir_name.rsplit(sep, 1)
            if len(parts) == 2:
                return parts[0], parts[1]

    return model_dir_name, "UnknownFeature"


def load_predictions_and_labels(csv_path):
    """Read predictions and labels from CSV file, return majority-vote prediction and labels."""
    if not os.path.exists(csv_path):
        return None, None

    try:
        df = pd.read_csv(csv_path)
        if (
            not set(PREDICTION_COLUMNS).issubset(df.columns)
            or LABEL_COLUMN not in df.columns
        ):
            print(f"Missing columns in {csv_path}")
            return None, None

        predictions = mode(df[PREDICTION_COLUMNS].values, axis=1, keepdims=False)[0]
        labels = df[LABEL_COLUMN].astype(int).values
        return predictions, labels

    except Exception as e:
        print(f"Failed to process {csv_path}: {e}")
        return None, None


def run_mcnemar_analysis():
    for isoform in ISOFORM_NAMES:
        print(f"\n--- Evaluating Isoform: {isoform} ---")
        mcnemar_results = {}

        isoform_path = os.path.join(BASE_DIR, isoform, SPLIT_TYPE)
        if not os.path.isdir(isoform_path):
            print(f"Path not found: {isoform_path}")
            continue

        model_dirs = [
            d
            for d in os.listdir(isoform_path)
            if os.path.isdir(os.path.join(isoform_path, d))
        ]
        model_info = {name: parse_model_name(name) for name in model_dirs}
        model_labels = {
            name: f"{alg}-{feat}" for name, (alg, feat) in model_info.items()
        }
        sorted_labels = sorted(set(model_labels.values()))

        for model1, model2 in itertools.combinations(model_dirs, 2):
            label1, label2 = model_labels[model1], model_labels[model2]
            print(f"  Comparing: {label1} vs {label2}")

            y_true_combined, pred1_combined, pred2_combined = [], [], []
            valid_data = True

            for fold in range(NUM_FOLDS):
                fold_name = f"{FOLD_PREFIX}{fold}"
                fold_path1 = os.path.join(isoform_path, model1, fold_name)
                fold_path2 = os.path.join(isoform_path, model2, fold_name)

                csv1 = f"test_{model_info[model1][0].lower()}_{model_info[model1][1].lower()}.csv"
                csv2 = f"test_{model_info[model2][0].lower()}_{model_info[model2][1].lower()}.csv"

                preds1, labels1 = load_predictions_and_labels(
                    os.path.join(fold_path1, csv1)
                )
                preds2, labels2 = load_predictions_and_labels(
                    os.path.join(fold_path2, csv2)
                )

                if any(x is None for x in (preds1, preds2, labels1, labels2)):
                    print(f"    Skipping fold {fold_name} due to missing data.")
                    valid_data = False
                    break

                if not np.array_equal(labels1, labels2):
                    print(
                        f"    Label mismatch in fold {fold_name}. Skipping comparison."
                    )
                    valid_data = False
                    break

                y_true_combined.extend(labels1)
                pred1_combined.extend(preds1)
                pred2_combined.extend(preds2)

            # Run McNemar if valid
            p_value = np.nan
            if valid_data and y_true_combined:
                y_true = np.array(y_true_combined)
                p1 = np.array(pred1_combined) == y_true
                p2 = np.array(pred2_combined) == y_true

                n11 = np.sum(p1 & p2)
                n10 = np.sum(p1 & ~p2)
                n01 = np.sum(~p1 & p2)
                n00 = np.sum(~p1 & ~p2)

                table = [[n11, n10], [n01, n00]]
                if (n10 + n01) == 0:
                    p_value = 1.0
                else:
                    try:
                        p_value = mcnemar(table, exact=True).pvalue
                    except ValueError as e:
                        print(f"    Error in McNemar test: {e}")

            # Store results in both directions
            mcnemar_results.setdefault(label1, {})[label2] = p_value
            mcnemar_results.setdefault(label2, {})[label1] = p_value

        # Fill diagonal and format
        df = pd.DataFrame.from_dict(
            mcnemar_results, orient="index", columns=sorted_labels
        )
        df = df.reindex(index=sorted_labels, columns=sorted_labels)
        for label in sorted_labels:
            df.at[label, label] = np.nan

        # Output
        print(f"\nMcNemar Test Results for {isoform}:")
        pd.set_option("display.width", 180)
        pd.set_option("display.max_columns", None)
        print(df)

        out_file = os.path.join(BASE_DIR, f"{isoform}_{SPLIT_TYPE}_mcnemar_results.csv")
        df.to_csv(out_file)
        print(f"Saved McNemar results to: {out_file}")

    print("\n✅ McNemar analysis complete for all isoforms.")


if __name__ == "__main__":
    run_mcnemar_analysis()
