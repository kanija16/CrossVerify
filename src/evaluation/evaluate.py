import sys
import os
import pandas as pd

from metrics import calculate_metrics


def evaluate_model(input_file, output_file=None):
    """
    Evaluate predictions stored in a CSV file.

    Required columns:
        true_label
        predicted_label
        probability
    """

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Prediction file not found: {input_file}"
        )

    # Read prediction file
    df = pd.read_csv(input_file)

    # Check required columns
    required_columns = [
        "true_label",
        "predicted_label",
        "probability"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # Calculate metrics
    results = calculate_metrics(df)

    # Display results
    print("\nCrossVerify Model Evaluation")
    print("-----------------------------")

    print(f"File       : {input_file}")
    print(f"Samples    : {len(df)}")

    print(
        f"Accuracy   : {results['accuracy']:.4f}"
    )

    print(
        f"Precision  : {results['precision']:.4f}"
    )

    print(
        f"Recall     : {results['recall']:.4f}"
    )

    print(
        f"F1-score   : {results['f1']:.4f}"
    )

    if results["auc"] is not None:
        print(
            f"ROC-AUC    : {results['auc']:.4f}"
        )
    else:
        print("ROC-AUC    : Not available")

    # Save results if output path is provided
    if output_file:

        results_df = pd.DataFrame([results])

        results_df.to_csv(
            output_file,
            index=False
        )

        print(
            f"\nResults saved to: {output_file}"
        )

    return results


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python evaluate.py <prediction_file> [output_file]"
        )

        print(
            "\nExample:"
        )

        print(
            "python evaluate.py results/cnn_predictions.csv results/cnn_metrics.csv"
        )

        sys.exit(1)

    input_file = sys.argv[1]

    output_file = None

    if len(sys.argv) >= 3:
        output_file = sys.argv[2]

    evaluate_model(
        input_file,
        output_file
    )
