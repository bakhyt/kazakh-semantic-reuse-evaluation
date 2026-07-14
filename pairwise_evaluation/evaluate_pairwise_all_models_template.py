"""
Pairwise PAN-KK evaluation template for all main models.

This script evaluates classifier, SBERT-style, and reranker-style models on
the PAN-KK pairwise semantic reuse test set.

Expected input CSV columns:
- kz_suspicious_text
- kz_source_text
- label

Private Google Drive paths, trained model folders, and dataset files are not
included in this repository.
"""

import argparse
import gc

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


TEXT_COLUMNS = ("kz_suspicious_text", "kz_source_text")
LABEL_COLUMN = "label"


def compute_metrics(y_true, y_pred):
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
    Load PAN-KK pairwise test data.
    """
    df = pd.read_csv(
        test_csv,
        usecols=[TEXT_COLUMNS[0], TEXT_COLUMNS[1], LABEL_COLUMN]
    ).dropna().reset_index(drop=True)

    suspicious_texts = df[TEXT_COLUMNS[0]].astype(str).tolist()
    source_texts = df[TEXT_COLUMNS[1]].astype(str).tolist()
    labels = df[LABEL_COLUMN].astype(int).to_numpy()

    return suspicious_texts, source_texts, labels


def evaluate_classifier(
    model_path,
    suspicious_texts,
    source_texts,
    labels,
    device,
    batch_size=32,
    max_length=512,
    trust_remote_code=False
):
    """
    Evaluate a sequence-classification model using paired tokenisation.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code
    ).to(device)

    model.eval()

    predictions = []

    for i in tqdm(
        range(0, len(labels), batch_size),
        desc="Classifier evaluation"
    ):
        batch_suspicious = suspicious_texts[i:i + batch_size]
        batch_source = source_texts[i:i + batch_size]

        encoded = tokenizer(
            batch_suspicious,
            batch_source,
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
            )

        predictions.extend(batch_predictions.tolist())

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return compute_metrics(labels, predictions)


def evaluate_sbert(
    model_path,
    suspicious_texts,
    source_texts,
    labels,
    device,
    threshold=0.60,
    batch_size=32
):
    """
    Evaluate an SBERT-style embedding model using cosine similarity.
    """
    model = SentenceTransformer(model_path, device=str(device))

    suspicious_embeddings = model.encode(
        suspicious_texts,
        convert_to_tensor=True,
        batch_size=batch_size,
        show_progress_bar=True
    )

    source_embeddings = model.encode(
        source_texts,
        convert_to_tensor=True,
        batch_size=batch_size,
        show_progress_bar=True
    )

    cosine_scores = (
        util.cos_sim(suspicious_embeddings, source_embeddings)
        .diagonal()
        .cpu()
        .numpy()
    )

    predictions = (cosine_scores >= threshold).astype(int)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    metrics = compute_metrics(labels, predictions)
    metrics["threshold"] = threshold

    return metrics


def evaluate_reranker(
    model_path,
    suspicious_texts,
    source_texts,
    labels,
    device,
    threshold,
    batch_size,
    max_length
):
    """
    Evaluate a CrossEncoder reranker model.
    """
    pairs = list(zip(suspicious_texts, source_texts))

    model = CrossEncoder(
        model_path,
        num_labels=1,
        max_length=max_length,
        trust_remote_code=True,
        device=str(device),
        model_kwargs={
            "torch_dtype": torch.bfloat16
            if torch.cuda.is_available()
            else torch.float32
        }
    )

    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=True
    )

    predictions = (np.array(scores) >= threshold).astype(int)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    metrics = compute_metrics(labels, predictions)
    metrics["threshold"] = threshold

    return metrics


def evaluate_all_models(test_csv: str, model_paths: dict, output_csv: str | None = None):
    """
    Evaluate all configured models and optionally save the result table.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    suspicious_texts, source_texts, labels = load_pairwise_data(test_csv)

    model_configs = {
        "XLM-R Large": {
            "type": "classifier",
            "path": model_paths["xlmr_large"],
            "max_length": 512,
            "batch_size": 32,
            "trust_remote_code": False,
        },
        "XLM-R Base": {
            "type": "classifier",
            "path": model_paths["xlmr_base"],
            "max_length": 512,
            "batch_size": 32,
            "trust_remote_code": False,
        },
        "DistilBERT": {
            "type": "classifier",
            "path": model_paths["distilbert"],
            "max_length": 512,
            "batch_size": 32,
            "trust_remote_code": True,
        },
        "MiniLM": {
            "type": "classifier",
            "path": model_paths["minilm"],
            "max_length": 512,
            "batch_size": 32,
            "trust_remote_code": True,
        },
        "BERTmulti": {
            "type": "classifier",
            "path": model_paths["bertmulti"],
            "max_length": 512,
            "batch_size": 32,
            "trust_remote_code": True,
        },
        "SBERT mpnet": {
            "type": "sbert",
            "path": model_paths["sbert_mpnet"],
            "threshold": 0.60,
            "batch_size": 32,
        },
        "Jina v3": {
            "type": "reranker",
            "path": model_paths["jina_v3"],
            "threshold": 0.55,
            "max_length": 512,
            "batch_size": 8,
        },
        "BGE Gemma": {
            "type": "reranker",
            "path": model_paths["bge_gemma"],
            "threshold": 0.35,
            "max_length": 1024,
            "batch_size": 4,
        },
    }

    results = {}

    for model_name, config in model_configs.items():
        print(f"Evaluating {model_name} ...")

        if config["type"] == "classifier":
            metrics = evaluate_classifier(
                model_path=config["path"],
                suspicious_texts=suspicious_texts,
                source_texts=source_texts,
                labels=labels,
                device=device,
                batch_size=config["batch_size"],
                max_length=config["max_length"],
                trust_remote_code=config["trust_remote_code"]
            )

        elif config["type"] == "sbert":
            metrics = evaluate_sbert(
                model_path=config["path"],
                suspicious_texts=suspicious_texts,
                source_texts=source_texts,
                labels=labels,
                device=device,
                threshold=config["threshold"],
                batch_size=config["batch_size"]
            )

        elif config["type"] == "reranker":
            metrics = evaluate_reranker(
                model_path=config["path"],
                suspicious_texts=suspicious_texts,
                source_texts=source_texts,
                labels=labels,
                device=device,
                threshold=config["threshold"],
                batch_size=config["batch_size"],
                max_length=config["max_length"]
            )

        else:
            raise ValueError(f"Unknown model type: {config['type']}")

        results[model_name] = metrics

    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df = results_df.sort_values(by="f1", ascending=False)

    if output_csv:
        results_df.to_csv(output_csv, index=True)

    return results_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate main models on pairwise PAN-KK data."
    )

    parser.add_argument("--test_csv", required=True)

    parser.add_argument("--xlmr_large_model_path", required=True)
    parser.add_argument("--xlmr_base_model_path", required=True)
    parser.add_argument("--distilbert_model_path", required=True)
    parser.add_argument("--minilm_model_path", required=True)
    parser.add_argument("--bertmulti_model_path", required=True)
    parser.add_argument("--sbert_mpnet_model_path", required=True)
    parser.add_argument("--jina_v3_model_path", required=True)
    parser.add_argument("--bge_gemma_model_path", required=True)

    parser.add_argument("--output_csv", default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_paths = {
        "xlmr_large": args.xlmr_large_model_path,
        "xlmr_base": args.xlmr_base_model_path,
        "distilbert": args.distilbert_model_path,
        "minilm": args.minilm_model_path,
        "bertmulti": args.bertmulti_model_path,
        "sbert_mpnet": args.sbert_mpnet_model_path,
        "jina_v3": args.jina_v3_model_path,
        "bge_gemma": args.bge_gemma_model_path,
    }

    result_table = evaluate_all_models(
        test_csv=args.test_csv,
        model_paths=model_paths,
        output_csv=args.output_csv
    )

    print("\nPairwise all-model results")
    print(result_table.to_string())
