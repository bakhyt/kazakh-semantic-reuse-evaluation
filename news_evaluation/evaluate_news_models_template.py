"""
News-domain semantic reuse evaluation template.

This script evaluates lexical and transformer-based models on the external
Kazakh news-domain snippet-pair benchmark.

Expected input CSV columns:
- nur_similar_text
- tengri_similar_text
- gold_binary_strict

The script does not include private model paths or copyrighted news text.
"""

import argparse
import gc
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SAFE_EMPTY = "[EMPTY]"

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


def safe_text(text: str, placeholder: str = SAFE_EMPTY) -> str:
    """
    Replace empty strings with a safe placeholder.
    """
    text = normalise_text(text)
    return text if text else placeholder


def split_dev_test(df: pd.DataFrame, random_state: int = 42):
    """
    Create 20% development and 80% test split using stratified sampling.
    """
    labels = df[GOLD_COLUMN].astype(int).values

    dev_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.8,
        random_state=random_state,
        stratify=labels
    )

    df_dev = df.iloc[dev_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    return df_dev, df_test


def evaluate_scores(name, scores_dev, scores_test, y_dev, y_test):
    """
    Select threshold on development set and evaluate on test set.
    """
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(0.10, 0.91, 0.05):
        pred_dev = (scores_dev >= threshold).astype(int)

        _, _, f1_dev, _ = precision_recall_fscore_support(
            y_dev,
            pred_dev,
            average="binary",
            zero_division=0
        )

        if f1_dev > best_f1:
            best_f1 = f1_dev
            best_threshold = threshold

    pred_test = (scores_test >= best_threshold).astype(int)

    accuracy = accuracy_score(y_test, pred_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        pred_test,
        average="binary",
        zero_division=0
    )

    return {
        "model": name,
        "threshold": round(float(best_threshold), 2),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def tfidf_char35_scores(df_in, col1, col2):
    """
    Compute TF-IDF character 3-5 cosine similarity scores.
    """
    scores = []

    for _, row in df_in.iterrows():
        text_a = safe_text(row[col1])
        text_b = safe_text(row[col2])

        try:
            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5)
            ).fit([text_a, text_b])

            matrix = vectorizer.transform([text_a, text_b])
            score = cosine_similarity(matrix[0], matrix[1])[0][0]

        except Exception:
            score = 0.0

        scores.append(float(score))

    return np.array(scores, dtype=float)


def classifier_scores(df_in, col1, col2, config, device):
    """
    Score text pairs using a sequence classification model.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"],
        trust_remote_code=config.get("trust_remote_code", False)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_path"],
        trust_remote_code=config.get("trust_remote_code", False),
        use_flash_attn=False if config.get("trust_remote_code", False) else None
    )

    if config["type"] == "jina_v2" and torch.cuda.is_available():
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)

    model.eval()

    texts1 = [safe_text(x) for x in df_in[col1].tolist()]
    texts2 = [safe_text(x) for x in df_in[col2].tolist()]

    scores_all = []
    batch_size = config["batch_size"]
    max_length = config["max_length"]

    for i in range(0, len(df_in), batch_size):
        batch_a = texts1[i:i + batch_size]
        batch_b = texts2[i:i + batch_size]

        encoded = tokenizer(
            batch_a,
            batch_b,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**encoded).logits

            if config["type"] == "jina_v2":
                scores = torch.sigmoid(logits.view(-1))
            elif logits.ndim == 2 and logits.shape[1] == 2:
                scores = torch.softmax(logits, dim=-1)[:, 1]
            else:
                scores = torch.sigmoid(logits.view(-1))

        scores_all.extend(scores.detach().float().cpu().numpy().tolist())

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return np.array(scores_all, dtype=float)


def crossencoder_scores(df_in, col1, col2, config):
    """
    Score text pairs using a sentence-transformers CrossEncoder.
    """
    texts1 = [safe_text(x) for x in df_in[col1].tolist()]
    texts2 = [safe_text(x) for x in df_in[col2].tolist()]
    pairs = list(zip(texts1, texts2))

    model = CrossEncoder(
        config["model_path"],
        num_labels=1,
        max_length=config["max_length"],
        trust_remote_code=config.get("trust_remote_code", False),
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs={
            "torch_dtype": torch.bfloat16
            if torch.cuda.is_available()
            else torch.float32
        }
    )

    scores_all = []
    batch_size = config["batch_size"]

    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]

        try:
            scores = model.predict(
                batch_pairs,
                batch_size=batch_size,
                show_progress_bar=False
            )
            scores_all.extend(np.asarray(scores, dtype=float).tolist())

        except Exception:
            scores_all.extend(np.zeros(len(batch_pairs), dtype=float).tolist())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return np.array(scores_all, dtype=float)


def evaluate_news_models(input_csv: str, output_csv: str | None = None):
    """
    Evaluate the news-domain benchmark.
    """
    df = pd.read_csv(input_csv)

    for column in TEXT_COLUMNS:
        df[column] = df[column].apply(safe_text)

    df_dev, df_test = split_dev_test(df)

    y_dev = df_dev[GOLD_COLUMN].astype(int).values
    y_test = df_test[GOLD_COLUMN].astype(int).values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_configs = {
        "TF-IDF char 3-5": {
            "type": "lexical"
        },
        "XLM-R Large": {
            "type": "classifier",
            "model_path": "path/to/xlm-r-large-model",
            "max_length": 512,
            "batch_size": 16,
            "trust_remote_code": False,
        },
        "XLM-R Base": {
            "type": "classifier",
            "model_path": "path/to/xlm-r-base-model",
            "max_length": 512,
            "batch_size": 16,
            "trust_remote_code": False,
        },
        "Jina v2 base multilingual": {
            "type": "jina_v2",
            "model_path": "path/to/jina-v2-base-multilingual-model",
            "max_length": 1024,
            "batch_size": 8,
            "trust_remote_code": True,
        },
        "BGE Gemma": {
            "type": "crossencoder",
            "model_path": "path/to/bge-reranker-v2-gemma-model",
            "max_length": 1024,
            "batch_size": 4,
            "trust_remote_code": True,
        },
    }

    results = []

    for model_name, config in model_configs.items():
        print(f"Running {model_name} ...")

        if config["type"] == "lexical":
            scores_dev = tfidf_char35_scores(df_dev, *TEXT_COLUMNS)
            scores_test = tfidf_char35_scores(df_test, *TEXT_COLUMNS)

        elif config["type"] in ["classifier", "jina_v2"]:
            scores_dev = classifier_scores(df_dev, *TEXT_COLUMNS, config, device)
            scores_test = classifier_scores(df_test, *TEXT_COLUMNS, config, device)

        elif config["type"] == "crossencoder":
            scores_dev = crossencoder_scores(df_dev, *TEXT_COLUMNS, config)
            scores_test = crossencoder_scores(df_test, *TEXT_COLUMNS, config)

        else:
            raise ValueError(f"Unknown model type: {config['type']}")

        result = evaluate_scores(
            model_name,
            scores_dev,
            scores_test,
            y_dev,
            y_test
        )

        print(result)
        results.append(result)

    results_df = pd.DataFrame(results)

    if output_csv:
        results_df.to_csv(output_csv, index=False)

    return results_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate models on the Kazakh news benchmark."
    )

    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to the news benchmark CSV."
    )

    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional path to save result table."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results_df = evaluate_news_models(
        input_csv=args.input_csv,
        output_csv=args.output_csv
    )

    print("\nNews-domain results")
    print(results_df.to_string(index=False))
