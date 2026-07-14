"""
Pairwise PAN-KK TF-IDF character 3-5 baseline evaluation.

This script evaluates a TF-IDF character n-gram baseline on suspicious-source
text pairs.

Expected input CSV columns:
- kz_suspicious_text
- kz_source_text
- label

The script does not redistribute dataset text or private file paths.
"""

import argparse
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def normalise_text(text: str) -> str:
    """
    Basic whitespace normalisation.
    """
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_tfidf_char35_scores(suspicious_texts, source_texts):
    """
    Compute TF-IDF character 3-5 cosine similarity scores.

    The vectorizer is fitted separately for each suspicious-source pair.
    """
    scores = []

    for suspicious, source in tqdm(
        zip(suspicious_texts, source_texts),
        total=len(suspicious_texts),
        desc="Scoring TF-IDF"
    ):
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5)
        )

        matrix = vectorizer.fit_transform([suspicious, source])
        score = cosine_similarity(matrix[0], matrix[1])[0][0]
        scores.append(float(score))

    return np.array(scores, dtype=float)


def evaluate_threshold(labels, scores, threshold):
    """
    Evaluate one threshold.
    """
    predictions = (scores >= threshold).astype(int)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    return accuracy, precision, recall, f1


def search_best_threshold(labels, scores, start=0.10, stop=0.90, step=0.05):
    """
    Search the best threshold by F1 score.

    Note: If this is applied directly to the test set, it should be described
    as a direct comparison or diagnostic setting, not as a strictly held-out
    model-selection procedure.
    """
    best_threshold = None
    best_f1 = -1.0
    best_metrics = None

    thresholds = np.arange(start, stop + step, step)

    for threshold in thresholds:
        accuracy, precision, recall, f1 = evaluate_threshold(
            labels,
            scores,
            threshold
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = accuracy, precision, recall, f1

    return best_threshold, best_metrics


def evaluate_tfidf_char35(
    test_csv: str,
    threshold: float | None = None,
    save_scores_path: str | None = None
):
    """
    Evaluate TF-IDF character 3-5 baseline on pairwise PAN-KK data.
    """
    df = pd.read_csv(
        test_csv,
        usecols=["kz_suspicious_text", "kz_source_text", "label"]
    ).dropna()

    df["kz_suspicious_text"] = df["kz_suspicious_text"].apply(normalise_text)
    df["kz_source_text"] = df["kz_source_text"].apply(normalise_text)

    suspicious_texts = df["kz_suspicious_text"].tolist()
    source_texts = df["kz_source_text"].tolist()
    labels = df["label"].astype(int).to_numpy()

    scores = compute_tfidf_char35_scores(
        suspicious_texts,
        source_texts
    )

    if threshold is None:
        threshold, metrics = search_best_threshold(labels, scores)
    else:
        metrics = evaluate_threshold(labels, scores, threshold)

    accuracy, precision, recall, f1 = metrics

    results = {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }

    if save_scores_path:
        score_df = pd.DataFrame({
            "label": labels,
            "tfidf_char35_score": scores,
            "tfidf_char35_pred": (scores >= threshold).astype(int)
        })
        score_df.to_csv(save_scores_path, index=False)

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TF-IDF char 3-5 on pairwise PAN-KK pairs."
    )

    parser.add_argument(
        "--test_csv",
        required=True,
        help="Path to the pairwise test CSV file."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold. If omitted, the script searches 0.10 to 0.90."
    )

    parser.add_argument(
        "--save_scores_path",
        default=None,
        help="Optional path to save labels, scores, and predictions only."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = evaluate_tfidf_char35(
        test_csv=args.test_csv,
        threshold=args.threshold,
        save_scores_path=args.save_scores_path
    )

    print("\nTF-IDF char 3-5 results")
    for metric, value in results.items():
        print(f"{metric:10s}: {value:.4f}")
