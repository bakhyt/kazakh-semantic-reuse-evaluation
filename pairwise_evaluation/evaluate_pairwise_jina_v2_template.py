"""
Pairwise PAN-KK evaluation template for Jina v2 base multilingual.

This script evaluates a binary semantic reuse classifier on suspicious-source
text pairs.

Expected input CSV columns:
- kz_suspicious_text
- kz_source_text
- label

The script does not include private model paths or dataset files.
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------------------
# Optional compatibility patch for some custom XLM-R based models.
# ---------------------------------------------------------------------
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


def evaluate_pairwise_jina_v2(
    model_path: str,
    test_csv: str,
    threshold: float = 0.60,
    batch_size: int = 8,
    max_length: int = 1024
):
    """
    Evaluate Jina v2 on pairwise PAN-KK suspicious-source pairs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    df = pd.read_csv(
        test_csv,
        usecols=["kz_suspicious_text", "kz_source_text", "label"]
    ).dropna()

    suspicious_texts = df["kz_suspicious_text"].tolist()
    source_texts = df["kz_source_text"].tolist()
    labels = df["label"].astype(int).tolist()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_flash_attn=False
    )

    if torch.cuda.is_available():
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device, dtype=torch.float32)

    model.eval()

    all_scores = []

    for i in tqdm(range(0, len(labels), batch_size), desc="Evaluating"):
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
            logits = model(**encoded).logits.view(-1)
            scores = torch.sigmoid(logits)
            all_scores.extend(scores.detach().float().cpu().numpy().tolist())

    predictions = (np.array(all_scores) >= threshold).astype(int)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    results = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "threshold": threshold,
        "batch_size": batch_size,
        "max_length": max_length
    }

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Jina v2 on pairwise PAN-KK pairs."
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the fine-tuned Jina v2 model."
    )

    parser.add_argument(
        "--test_csv",
        required=True,
        help="Path to the pairwise test CSV file."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Decision threshold."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Evaluation batch size."
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum transformer input length."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = evaluate_pairwise_jina_v2(
        model_path=args.model_path,
        test_csv=args.test_csv,
        threshold=args.threshold,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    print("\nResults")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric:12s}: {value:.4f}")
        else:
            print(f"{metric:12s}: {value}")
