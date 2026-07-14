import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def evaluate_predictions(input_csv: str):
    """
    Evaluate binary pairwise semantic reuse predictions.

    Expected CSV columns:
    - gold_label: 0 or 1
    - predicted_label: 0 or 1
    """
    df = pd.read_csv(input_csv)

    y_true = df["gold_label"]
    y_pred = df["predicted_label"]

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    return results


if __name__ == "__main__":
    # Example:
    # python evaluate_pairwise_predictions.py predictions.csv
    import sys

    if len(sys.argv) != 2:
        print("Usage: python evaluate_pairwise_predictions.py predictions.csv")
        sys.exit(1)

    results = evaluate_predictions(sys.argv[1])

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
