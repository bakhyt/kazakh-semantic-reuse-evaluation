"""
Pairwise PAN-KK evaluation template for additional models.

This script evaluates additional pairwise semantic reuse models on the PAN-KK
pairwise test set.

Models covered:
- DistilBERT
- MiniLM
- BERTmulti
- SBERT mpnet
- TF-IDF char 3-5 baseline

Expected input CSV columns:
- kz_suspicious_text
- kz_source_text
- label

Private Google Drive paths, trained model folders, and dataset files are not
included in this repository.
"""

import argparse
import gc
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


TEXT_COLUMNS = ("kz_suspicious_text", "kz_source_text")
LABEL_COLUMN = "label"


# Optional compatibility patch for XLM-R based models.
import transformers.models.xlm_roberta.modeling_xlm_roberta as xlm_mod

if not hasattr(xlm_mod, "create_position_ids_from_input_ids"):
    def create_position_ids_from_input_ids(
        input_ids,
        padding_idx,
        past_key_values_length=0
    ):
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (
            torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
        ) * mask
        return incremental_indices.long() + padding_idx

    xlm_mod.create_position_ids_from_input_ids = create_position_ids_from_input_ids


def normalise_text(text: str) -> str:
    """
    Basic whitespace normalisation.
    """
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_metrics(y_true, y_pred):
    """
    Compute accuracy, precision, recall, and F1.
    """
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    return {
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
    }


def load_pairwise_data(test_csv: str):
    """
    Load pairwise PAN-KK test data.
    """
    df = pd.read_csv(
        test_csv,
        usecols=[TEXT_COLUMNS[0], TEXT_COLUMNS[1], LABEL_COLUMN]
    ).dropna()

    df[TEXT_COLUMNS[0]] = df[TEXT_COLUMNS[0]].apply(normalise_text)
    df[TEXT_COLUMNS[1]] = df[TEXT_COLUMNS[1]].apply(normalise_text)

    suspicious_texts = df[TEXT_COLUMNS[0]].tolist()
    source_texts = df[TEXT_COLUMNS[1]].tolist()
    labels = df[LABEL_COLUMN].astype(int).to_numpy()

    return suspicious_texts, source_texts, labels


def evaluate_classifier(
    model_name: str,
    model_path: str,
    suspicious_texts,
    source_texts,
    labels,
    device,
    batch_size: int = 32,
    max_length: int = 512
):
    """
    Evaluate a sequence-classification model using argmax prediction.
    """
    print(f"Evaluating {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True
    ).to(device)

    model.eval()

    predictions = []

    for i in tqdm(
        range(0, len(labels), batch_size),
        desc=f"Classifier: {model_name}"
    ):
        batch_suspicious = suspicious_texts[i:i + batch_size]
        batch_source = source_texts[i:i + batch_size]

        encoded = tokenizer(
            batch_suspicious,
            text_pair=batch_source,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**encoded).logits
            batch_predictions = (
                torch.argmax(logits, dim=-1)
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

        predictions.extend(batch_predictions)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return calculate_metrics(labels, predictions)


def evaluate_sbert_mpnet(
    model_path: str,
    suspicious_texts,
    source_texts,
    labels,
    device,
    threshold: float = 0.60,
    batch_size: int = 32
):
    """
    Evaluate SBERT mpnet using cosine similarity between sentence embeddings.
    """
    print("Evaluating SBERT mpnet ...")

    model = SentenceTransformer(model_path, device=str(device))

    suspicious_embeddings = model.encode(
        suspicious_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    source_embeddings = model.encode(
        source_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities = (suspicious_embeddings * source_embeddings).sum(axis=1)
    predictions = (similarities >= threshold).astype(int)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    metrics = calculate_metrics(labels, predictions)
    metrics["threshold"] = threshold

    return metrics


def evaluate_tfidf_char35(
    suspicious_texts,
    source_texts,
    labels,
    threshold: float | None = None
):
    """
    Evaluate TF-IDF character 3-5 cosine similarity baseline.

    If no threshold is supplied, the best threshold is selected by grid search
    on the provided data. This should be described as a direct comparison or
    diagnostic baseline, not as strict held-out model selection.
    """
    print("Evaluating TF-IDF char 3-5 ...")

    scores = []

    for suspicious, source in tqdm(
        zip(suspicious_texts, source_texts),
        total=len(labels),
        desc="TF-IDF char 3-5"
    ):
        try:
            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5)
            )

            matrix = vectorizer.fit_transform([suspicious, source])
            score = cosine_similarity(matrix[0], matrix[1])[0][0]

        except Exception:
            score = 0.0

        scores.append(float(score))

    scores = np.array(scores, dtype=float)

    if threshold is None:
        best_threshold = 0.5
        best_f1 = -1.0
        best_predictions = None

        for current_threshold in np.arange(0.10, 0.91, 0.05):
            predictions = (scores >= current_threshold).astype(int)
            metrics = calculate_metrics(labels, predictions)

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = current_threshold
                best_predictions = predictions

        metrics = calculate_metrics(labels, best_predictions)
        metrics["best_threshold"] = round(float(best_threshold), 2)

    else:
        predictions = (scores >= threshold).astype(int)
        metrics = calculate_metrics(labels, predictions)
        metrics["threshold"] = threshold

    return metrics


def evaluate_additional_models(
    test_csv: str,
    distilbert_model_path: str,
    minilm_model_path: str,
    bertmulti_model_path: str,
    sbert_mpnet_model_path: str,
    output_csv: str | None = None
):
    """
    Evaluate additional pairwise models and optionally save result table.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    suspicious_texts, source_texts, labels = load_pairwise_data(test_csv)

    results = {}

    results["DistilBERT"] = evaluate_classifier(
        model_name="DistilBERT",
        model_path=distilbert_model_path,
        suspicious_texts=suspicious_texts,
        source_texts=source_texts,
        labels=labels,
        device=device,
        batch_size=32,
        max_length=512
    )

    results["MiniLM"] = evaluate_classifier(
        model_name="MiniLM",
        model_path=minilm_model_path,
        suspicious_texts=suspicious_texts,
        source_texts=source_texts,
        labels=labels,
        device=device,
        batch_size=32,
        max_length=512
    )

    results["BERTmulti"] = evaluate_classifier(
        model_name="BERTmulti",
        model_path=bertmulti_model_path,
        suspicious_texts=suspicious_texts,
        source_texts=source_texts,
        labels=labels,
        device=device,
        batch_size=32,
        max_length=512
    )

    results["SBERT_mpnet"] = evaluate_sbert_mpnet(
        model_path=sbert_mpnet_model_path,
        suspicious_texts=suspicious_texts,
        source_texts=source_texts,
        labels=labels,
        device=device,
        threshold=0.60,
        batch_size=32
    )

    results["TF-IDF char 3-5"] = evaluate_tfidf_char35(
        suspicious_texts=suspicious_texts,
        source_texts=source_texts,
        labels=labels,
        threshold=0.45
    )

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values("f1", ascending=False)

    if output_csv:
        results_df.to_csv(output_csv)

    return results_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate additional models on pairwise PAN-KK data."
    )

    parser.add_argument(
        "--test_csv",
        required=True,
        help="Path to the pairwise PAN-KK test CSV."
    )

    parser.add_argument(
        "--distilbert_model_path",
        required=True,
        help="Path to the DistilBERT model."
    )

    parser.add_argument(
        "--minilm_model_path",
        required=True,
        help="Path to the MiniLM model."
    )

    parser.add_argument(
        "--bertmulti_model_path",
        required=True,
        help="Path to the BERTmulti model."
    )

    parser.add_argument(
        "--sbert_mpnet_model_path",
        required=True,
        help="Path to the SBERT mpnet model."
    )

    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional path to save results."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result_table = evaluate_additional_models(
        test_csv=args.test_csv,
        distilbert_model_path=args.distilbert_model_path,
        minilm_model_path=args.minilm_model_path,
        bertmulti_model_path=args.bertmulti_model_path,
        sbert_mpnet_model_path=args.sbert_mpnet_model_path,
        output_csv=args.output_csv
    )

    print("\nPairwise additional model results")
    print(result_table.to_string())
