"""
News-domain evaluation template for additional models.

This script evaluates additional models on the external Kazakh news-domain
benchmark, including DistilBERT, MiniLM, BERTmulti, and SBERT mpnet.

Expected input CSV columns:
- nur_similar_text
- tengri_similar_text
- gold_binary_strict

The script does not include private Google Drive paths, trained model files,
or copyrighted news text.
"""

import argparse
import gc
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


TEXT_COLUMNS = ("nur_similar_text", "tengri_similar_text")
GOLD_COLUMN = "gold_binary_strict"


# Optional compatibility patch for some custom XLM-R based models.
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
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def evaluate_classifier(
    model_name: str,
    model_path: str,
    df_split: pd.DataFrame,
    device,
    batch_size: int = 32,
    max_length: int = 512
):
    """
    Evaluate a sequence-classification model.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True
    ).to(device)

    model.eval()

    texts1 = df_split[TEXT_COLUMNS[0]].tolist()
    texts2 = df_split[TEXT_COLUMNS[1]].tolist()

    predictions = []

    for i in tqdm(
        range(0, len(df_split), batch_size),
        desc=f"Classifier: {model_name}"
    ):
        batch_1 = texts1[i:i + batch_size]
        batch_2 = texts2[i:i + batch_size]

        encoded = tokenizer(
            batch_1,
            text_pair=batch_2,
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

    return np.array(predictions)


def evaluate_sbert_with_dev_tuning(
    model_path: str,
    df_dev: pd.DataFrame,
    y_dev,
    df_test: pd.DataFrame,
    y_test,
    device,
    batch_size: int = 32
):
    """
    Evaluate an SBERT-style embedding model using cosine similarity.

    Threshold is selected on the development split and evaluated on the
    held-out test split.
    """
    model = SentenceTransformer(model_path, device=str(device))

    dev_1 = df_dev[TEXT_COLUMNS[0]].tolist()
    dev_2 = df_dev[TEXT_COLUMNS[1]].tolist()
    test_1 = df_test[TEXT_COLUMNS[0]].tolist()
    test_2 = df_test[TEXT_COLUMNS[1]].tolist()

    emb_dev_1 = model.encode(
        dev_1,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    emb_dev_2 = model.encode(
        dev_2,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dev_scores = (emb_dev_1 * emb_dev_2).sum(axis=1)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(0.10, 0.91, 0.05):
        dev_predictions = (dev_scores >= threshold).astype(int)
        metrics = calculate_metrics(y_dev, dev_predictions)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold

    emb_test_1 = model.encode(
        test_1,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    emb_test_2 = model.encode(
        test_2,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    test_scores = (emb_test_1 * emb_test_2).sum(axis=1)
    test_predictions = (test_scores >= best_threshold).astype(int)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    metrics = calculate_metrics(y_test, test_predictions)
    metrics["threshold"] = round(float(best_threshold), 2)

    return metrics


def evaluate_news_additional_models(
    input_csv: str,
    model_paths: dict,
    output_csv: str | None = None
):
    """
    Evaluate additional models on the news-domain benchmark.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(input_csv)

    for column in TEXT_COLUMNS:
        df[column] = df[column].apply(normalise_text)

    df = df[
        (df[TEXT_COLUMNS[0]] != "")
        & (df[TEXT_COLUMNS[1]] != "")
    ].reset_index(drop=True)

    labels = df[GOLD_COLUMN].astype(int).values

    dev_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.8,
        random_state=42,
        stratify=labels
    )

    df_dev = df.iloc[dev_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    y_dev = df_dev[GOLD_COLUMN].astype(int).values
    y_test = df_test[GOLD_COLUMN].astype(int).values

    results = {}

    for model_name in ["DistilBERT", "MiniLM", "BERTmulti"]:
        print(f"Evaluating {model_name} ...")

        predictions = evaluate_classifier(
            model_name=model_name,
            model_path=model_paths[model_name],
            df_split=df_test,
            device=device,
            batch_size=32,
            max_length=512
        )

        results[model_name] = calculate_metrics(y_test, predictions)

    print("Evaluating SBERT_mpnet ...")

    results["SBERT_mpnet"] = evaluate_sbert_with_dev_tuning(
        model_path=model_paths["SBERT_mpnet"],
        df_dev=df_dev,
        y_dev=y_dev,
        df_test=df_test,
        y_test=y_test,
        device=device,
        batch_size=32
    )

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values("f1", ascending=False)

    if output_csv:
        results_df.to_csv(output_csv)

    return results_df


def parse_model_paths(args):
    """
    Collect model paths from command-line arguments.
    """
    return {
        "DistilBERT": args.distilbert_model_path,
        "MiniLM": args.minilm_model_path,
        "BERTmulti": args.bertmulti_model_path,
        "SBERT_mpnet": args.sbert_mpnet_model_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate additional models on the Kazakh news benchmark."
    )

    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to the news benchmark CSV."
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

    model_paths = parse_model_paths(args)

    results_df = evaluate_news_additional_models(
        input_csv=args.input_csv,
        model_paths=model_paths,
        output_csv=args.output_csv
    )

    print("\nAdditional news-domain results")
    print(results_df.to_string())
