import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def convert_ordinal_to_binary(label: int) -> int:
    """
    Converts the 4-point news annotation label to binary.

    0--1 = non-similar
    2--3 = similar
    """
    if label in [0, 1]:
        return 0
    if label in [2, 3]:
        return 1

    raise ValueError(f"Unexpected label: {label}")


def evaluate_news_predictions(input_csv: str):
    """
    Evaluate news-domain semantic reuse predictions.

    Expected CSV columns:
    - gold_label: ordinal label 0, 1, 2, or 3
    - predicted_label: binary prediction 0 or 1
    """
    df = pd.read_csv(input_csv)

    y_true = df["gold_label"].apply(convert_ordinal_to_binary)
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
    import sys

    if len(sys.argv) != 2:
        print("Usage: python evaluate_news_predictions.py predictions.csv")
        sys.exit(1)

    results = evaluate_news_predictions(sys.argv[1])

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
