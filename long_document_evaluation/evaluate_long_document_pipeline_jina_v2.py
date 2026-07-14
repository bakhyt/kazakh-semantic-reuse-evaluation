import argparse
import datetime
import gc
import os
import re
import time

import pandas as pd
import torch
import stanza
from simhash import Simhash
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# Patch for Jina custom XLM-R code
# =========================================================
import transformers.models.xlm_roberta.modeling_xlm_roberta as xlm_mod

if not hasattr(xlm_mod, "create_position_ids_from_input_ids"):
    def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (
            torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
        ) * mask
        return incremental_indices.long() + padding_idx

    xlm_mod.create_position_ids_from_input_ids = create_position_ids_from_input_ids


# =========================================================
# Configuration
# =========================================================

VERIFIER_CONFIG = {
    "xlmr_large": {
        "type": "classifier",
        "threshold": 0.50,
        "batch_size": 8,
        "max_length": 512,
    },
    "xlmr_base": {
        "type": "classifier",
        "threshold": 0.50,
        "batch_size": 8,
        "max_length": 512,
    },
    "jina_v2_base_multi": {
        "type": "jina_v2_classifier",
        "threshold": 0.60,
        "batch_size": 8,
        "max_length": 1024,
    },
}


SIMHASH_NGRAM_SIZE = 1
SHORT_SENTENCE_K = 24
LONG_SENTENCE_K = 31
MIN_SENTENCE_TOKEN_LENGTH = 2
LEXICAL_SIMILARITY_THRESHOLD = 0.10


# =========================================================
# Preprocessing
# =========================================================

def preprocess_kazakh_text(text: str) -> str:
    """
    Basic Kazakh text normalisation before sentence segmentation,
    lemmatisation, candidate retrieval, and token-level evaluation.
    """
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s.!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str, nlp, min_token_len: int = MIN_SENTENCE_TOKEN_LENGTH):
    """
    Split text into Kazakh sentences using Stanza.
    Very short sentences are removed.
    """
    doc = nlp(preprocess_kazakh_text(text))
    return [
        sentence.text
        for sentence in doc.sentences
        if len(sentence.words) >= min_token_len
    ]


def lemmatize_text(text: str, nlp, cache: dict) -> str:
    """
    Lemmatise Kazakh text using Stanza.
    Results are cached to avoid repeated processing.
    """
    if text in cache:
        return cache[text]

    doc = nlp(preprocess_kazakh_text(text))
    result = " ".join(
        word.lemma if word.lemma else word.text
        for sent in doc.sentences
        for word in sent.words
    )

    cache[text] = result
    return result


# =========================================================
# Candidate retrieval
# =========================================================

def get_shingles(text: str, n: int):
    """
    Create token-based n-gram shingles.
    In the reported long-document setting, n = 1.
    """
    words = text.split()
    if len(words) < n:
        return [text]
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def compute_simhash(text: str, n: int, cache: dict):
    """
    Compute SimHash fingerprint for text.
    """
    if text in cache:
        return cache[text]

    value = Simhash(get_shingles(text, n)).value
    cache[text] = value
    return value


def lexical_similarity(text1: str, text2: str) -> float:
    """
    Compute TF-IDF cosine similarity between two text strings.
    """
    vectorizer = TfidfVectorizer().fit([text1, text2])
    vectors = vectorizer.transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]


def generate_candidate_pairs(
    suspicious_sentences,
    source_sentences,
    nlp,
    lemmatize_cache,
    simhash_cache,
):
    """
    Generate candidate suspicious-source sentence pairs using:
    1. exact SimHash match
    2. adaptive SimHash distance over lemmatised text
    3. TF-IDF lexical similarity threshold
    """
    exact_matches = []
    candidate_pairs = []

    for suspicious_text in suspicious_sentences:
        suspicious_hash = compute_simhash(
            suspicious_text,
            SIMHASH_NGRAM_SIZE,
            simhash_cache
        )

        match_found = False

        for source_text in source_sentences:
            source_hash = compute_simhash(
                source_text,
                SIMHASH_NGRAM_SIZE,
                simhash_cache
            )

            distance = bin(suspicious_hash ^ source_hash).count("1")

            if distance == 0:
                exact_matches.append((suspicious_text, 1.0))
                match_found = True
                break

        if match_found:
            continue

        suspicious_lemma = lemmatize_text(suspicious_text, nlp, lemmatize_cache)

        for source_text in source_sentences:
            source_lemma = lemmatize_text(source_text, nlp, lemmatize_cache)

            suspicious_lemma_hash = compute_simhash(
                suspicious_lemma,
                SIMHASH_NGRAM_SIZE,
                simhash_cache
            )
            source_lemma_hash = compute_simhash(
                source_lemma,
                SIMHASH_NGRAM_SIZE,
                simhash_cache
            )

            distance = bin(suspicious_lemma_hash ^ source_lemma_hash).count("1")

            adaptive_k = (
                SHORT_SENTENCE_K
                if len(suspicious_lemma.split()) < 10
                else LONG_SENTENCE_K
            )

            if distance == 0:
                exact_matches.append((suspicious_text, 1.0))
                match_found = True
                break

            if (
                distance < adaptive_k
                and lexical_similarity(suspicious_lemma, source_lemma)
                > LEXICAL_SIMILARITY_THRESHOLD
            ):
                candidate_pairs.append((suspicious_text, source_text, distance))

    return exact_matches, candidate_pairs


# =========================================================
# Verifier model
# =========================================================

def load_verifier(verifier_name: str, model_path: str, device):
    """
    Load the selected verifier model and tokenizer.
    """
    config = VERIFIER_CONFIG[verifier_name]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_flash_attn=False,
    )

    if config["type"] == "jina_v2_classifier":
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = model.to(device=device, dtype=dtype)
    else:
        model = model.to(device)

    model.eval()

    return tokenizer, model, config


def classify_candidate_pairs(pairs, tokenizer, model, config, device):
    """
    Apply the verifier model to candidate sentence pairs.
    """
    if not pairs:
        return []

    results = []
    batch_size = config["batch_size"]
    threshold = config["threshold"]
    max_length = config["max_length"]

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        batch_suspicious = [suspicious for suspicious, source, _ in batch]
        batch_source = [source for suspicious, source, _ in batch]

        inputs = tokenizer(
            batch_suspicious,
            text_pair=batch_source,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

            if config["type"] == "jina_v2_classifier":
                probabilities = torch.sigmoid(logits.view(-1)).detach().float().cpu().numpy()
            else:
                probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

        for j, score in enumerate(probabilities):
            if score >= threshold:
                results.append((batch[j][0], float(score)))

        del inputs, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# =========================================================
# Metrics
# =========================================================

def compute_token_metrics(gold_tokens, predicted_tokens):
    """
    Compute token-level precision, recall, and F1 using set overlap.
    """
    gold_set = set(gold_tokens)
    predicted_set = set(predicted_tokens)

    true_positive = len(gold_set & predicted_set)
    false_positive = len(predicted_set - gold_set)
    false_negative = len(gold_set - predicted_set)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return round(precision, 5), round(recall, 5), round(f1, 5)


# =========================================================
# Main evaluation
# =========================================================

def evaluate_long_document_pipeline(
    input_csv: str,
    model_path: str,
    output_dir: str,
    verifier_name: str = "jina_v2_base_multi",
    use_gpu_stanza: bool = True,
):
    """
    Evaluate the long-document semantic reuse detection pipeline.

    Expected CSV columns:
    - suspicious_document
    - source_document
    - suspicious_content
    - source_content
    - suspicious_words

    The public version does not save full gold or predicted text
    in the error-analysis file to avoid redistributing dataset content.
    """
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nlp = stanza.Pipeline(
        "kk",
        processors="tokenize,lemma",
        use_gpu=use_gpu_stanza and torch.cuda.is_available()
    )

    tokenizer, model, config = load_verifier(verifier_name, model_path, device)

    lemmatize_cache = {}
    simhash_cache = {}

    df = pd.read_csv(input_csv)

    results = []
    error_analysis = []

    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    for index, row in df.iterrows():
        print(
            f"Processing {index + 1}/{len(df)}: "
            f"{row['suspicious_document']} vs {row['source_document']}"
        )

        if pd.isna(row["suspicious_content"]) or pd.isna(row["source_content"]):
            continue

        suspicious_sentences = split_into_sentences(
            str(row["suspicious_content"]),
            nlp
        )
        source_sentences = split_into_sentences(
            str(row["source_content"]),
            nlp
        )

        simhash_matches, candidate_pairs = generate_candidate_pairs(
            suspicious_sentences,
            source_sentences,
            nlp,
            lemmatize_cache,
            simhash_cache,
        )

        verifier_matches = classify_candidate_pairs(
            candidate_pairs,
            tokenizer,
            model,
            config,
            device,
        )

        predicted_text = " ".join(
            text for text, _ in simhash_matches + verifier_matches
        )

        predicted_tokens = lemmatize_text(
            predicted_text,
            nlp,
            lemmatize_cache
        ).split()

        gold_tokens = lemmatize_text(
            str(row["suspicious_words"]),
            nlp,
            lemmatize_cache
        ).split()

        precision, recall, f1 = compute_token_metrics(
            gold_tokens,
            predicted_tokens
        )

        results.append({
            "verifier": verifier_name,
            "suspicious_document": row["suspicious_document"],
            "source_document": row["source_document"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        error_analysis.append({
            "verifier": verifier_name,
            "suspicious_document": row["suspicious_document"],
            "source_document": row["source_document"],
            "gold_token_count": len(gold_tokens),
            "predicted_token_count": len(predicted_tokens),
            "common_token_count": len(set(gold_tokens) & set(predicted_tokens)),
            "missing_token_count": len(set(gold_tokens) - set(predicted_tokens)),
            "extra_token_count": len(set(predicted_tokens) - set(gold_tokens)),
        })

    evaluation_df = pd.DataFrame(results)
    error_df = pd.DataFrame(error_analysis)

    metrics_path = os.path.join(
        output_dir,
        f"{verifier_name}_evaluation_metrics_{timestamp}.csv"
    )
    errors_path = os.path.join(
        output_dir,
        f"{verifier_name}_evaluation_errors_{timestamp}.csv"
    )

    evaluation_df.to_csv(metrics_path, index=False)
    error_df.to_csv(errors_path, index=False)

    print("Saved evaluation metrics to:", metrics_path)
    print("Saved error analysis to:", errors_path)

    print("\nAverage Metrics:")
    print(f"Precision: {evaluation_df['precision'].mean():.4f}")
    print(f"Recall:    {evaluation_df['recall'].mean():.4f}")
    print(f"F1 Score:  {evaluation_df['f1'].mean():.4f}")

    print(f"Total time: {time.time() - start_time:.2f} seconds")

    del df, results, error_analysis, evaluation_df, error_df
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the long-document semantic reuse detection pipeline."
    )

    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to the input CSV file."
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the verifier model."
    )

    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory where evaluation outputs will be saved."
    )

    parser.add_argument(
        "--verifier",
        default="jina_v2_base_multi",
        choices=list(VERIFIER_CONFIG.keys()),
        help="Verifier model configuration to use."
    )

    parser.add_argument(
        "--no_gpu_stanza",
        action="store_true",
        help="Disable GPU use for Stanza."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate_long_document_pipeline(
        input_csv=args.input_csv,
        model_path=args.model_path,
        output_dir=args.output_dir,
        verifier_name=args.verifier,
        use_gpu_stanza=not args.no_gpu_stanza,
    )
