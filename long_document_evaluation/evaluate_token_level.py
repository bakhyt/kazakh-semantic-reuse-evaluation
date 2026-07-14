import pandas as pd


def tokenise(text: str):
    """
    Simple whitespace tokenisation for token-level overlap evaluation.
    The final paper should describe the exact tokenisation used in the experiments.
    """
    if not isinstance(text, str):
        return set()

    return set(text.lower().split())


def calculate_token_metrics(gold_text: str, predicted_text: str):
    """
    Calculate token-level precision, recall, and F1.
    """
    gold_tokens = tokenise(gold_text)
    predicted_tokens = tokenise(predicted_text)

    if len(predicted_tokens) == 0:
        precision = 0.0
    else:
        precision = len(gold_tokens & predicted_tokens) / len(predicted_tokens)

    if len(gold_tokens) == 0:
        recall = 0.0
    else:
        recall = len(gold_tokens & predicted_tokens) / len(gold_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def evaluate_long_document_predictions(input_csv: str):
    """
    Expected CSV columns:
    - gold_reused_text
    - predicted_reused_text
    """
    df = pd.read_csv(input_csv)

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for _, row in df.iterrows():
        precision, recall, f1 = calculate_token_metrics(
            row["gold_reused_text"],
            row["predicted_reused_text"]
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    results = {
        "precision": sum(precision_scores) / len(precision_scores),
        "recall": sum(recall_scores) / len(recall_scores),
        "f1": sum(f1_scores) / len(f1_scores)
    }

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python evaluate_token_level.py long_document_predictions.csv")
        sys.exit(1)

    results = evaluate_long_document_predictions(sys.argv[1])

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
