import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def calculate_metrics(df):
    """
    Calculate classification performance metrics.

    Required columns in df:
        true_label
        predicted_label
        probability
    """

    y_true = df["true_label"]
    y_pred = df["predicted_label"]
    y_prob = df["probability"]

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = None

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


def evaluate_prediction_file(input_file, output_file=None):
    """
    Read a prediction CSV file and calculate evaluation metrics.
    """

    df = pd.read_csv(input_file)

    required_columns = [
        "true_label",
        "predicted_label",
        "probability"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    results = calculate_metrics(df)

    results_df = pd.DataFrame([results])

    if output_file:
        results_df.to_csv(
            output_file,
            index=False
        )

    return results


if __name__ == "__main__":

    print("CrossVerify Evaluation Module")
    print("--------------------------------")

    print("This module calculates:")
    print("- Accuracy")
    print("- Precision")
    print("- Recall")
    print("- F1-score")
    print("- ROC-AUC")
