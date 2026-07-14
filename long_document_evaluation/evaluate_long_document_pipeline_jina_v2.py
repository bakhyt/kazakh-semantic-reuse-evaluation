"""
Long-document semantic reuse evaluation template.

This script documents the long-document evaluation pipeline used for the
CLEF 2026 Kazakh semantic reuse detection experiments.

The script is intentionally written as a reproducible template. Local model
paths, dataset paths, and full copyrighted text are not included in this
repository.

Expected input CSV columns:
- suspicious_document
- source_document
- suspicious_content
- source_content
- suspicious_words

The column suspicious_words should contain the gold reused text or gold reused
tokens for the suspicious document, depending on the evaluation setup.
"""

import argparse
import datetime
import gc
import os
import re
import time

import pandas as pd
import stanza
import torch
from simhash import Simhash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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


# ---------------------------------------------------------------------
# Verifier configuration.
# Replace model_path values with local paths or model identifiers.
# ---------------------------------------------------------------------
VERIFIER_CONFIG = {
    "xlmr_large": {
        "type": "classifier",
        "model_path": "path/to/kz-xlm-roberta-large-model",
        "threshold": 0.50,
        "batch_size": 8,
        "max_length": 512,
    },
    "xlmr_base": {
        "type": "classifier",
        "model_path": "path/to/kz-xlm-roberta-base-model",
        "threshold": 0.50,
        "batch_size": 8,
        "max_length": 512,
    },
    "jina_v2_base_multi": {
        "type": "jina_v2_classifier",
        "model_path": "path/to/jina-v2-base-multilingual-model",
        "threshold": 0.60,
        "batch_size": 8,
        "max_length": 1024,
    },
}


# ---------------------------------------------------------------------
# Candidate retrieval configuration.
# ---------------------------------------------------------------------
SIMHASH_NGRAM_SIZE = 1
SHORT_SENTENCE_K = 24
LONG_SENTENCE_K = 31
SHORT_SENTENCE_TOKEN_LIMIT = 10
MIN_SENTENCE_TOKEN_LENGTH = 2
LEXICAL_SIMILARITY_THRESHOLD = 0.10


lemmatize_text_cache = {}
simhash_cache = {}


def preprocess_kazakh_text(text: str) -> str:
    """
    Basic Kazakh text normalisation before sentence splitting,
    lemmatisation, SimHash, and TF-IDF comparison.
    """
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s.!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str, nlp) -> list[str]:
    """
    Split text into Kazakh sentences using Stanza.
    Very short sentences are removed.
    """
    doc = nlp(preprocess_kazakh_text(text))
    return [
        sentence.text
        for sentence in doc.sentences
        if len(sentence.words) >= MIN_SENTENCE_TOKEN_LENGTH
    ]


def lemmatize_text(text: str, nlp) -> str:
    """
    Lemmatise Kazakh text using Stanza and cache the result.
    """
    if text in lemmatize_text_cache:
        return lemmatize_text_cache[text]

    doc = nlp(preprocess_kazakh_text(text))
    result = " ".join(
        word.lemma if word.lemma else word.text
        for sent in doc.sentences
        for word in sent.words
    )

    lemmatize_text_cache[text] = result
    return result


def get_shingles(text: str, n: int) -> list[str]:
    """
    Create token-based n-gram shingles for SimHash.
    In the final paper, this should be described as token-unigram
    SimHash when n = 1.
    """
    words = text.split()

    if len(words) < n:
        return [text]

    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def compute_simhash(text: str, n: int) -> int:
    """
    Compute SimHash fingerprint for a text string.
    """
    cache_key = (text, n)

    if cache_key in simhash_cache:
        return simhash_cache[cache_key]

    value = Simhash(get_shingles(text, n)).value
    simhash_cache[cache_key] = value
    return value


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """
    Compute Hamming distance between two SimHash fingerprints.
    """
    return bin(hash_a ^ hash_b).count("1")


def lexical_similarity(text1: str, text2: str) -> float:
    """
    Compute TF-IDF cosine similarity between two texts.
    """
    vectorizer = TfidfVectorizer().fit([text1, text2])
    vectors = vectorizer.transform([text1, text2])
    return float(cosine_similarity(vectors[0], vectors[1])[0][0])


def generate_candidate_pairs(
    suspicious_sentences: list[str],
    source_sentences: list[str],
    nlp
):
    """
    Generate exact SimHash matches and candidate sentence pairs.

    Exact matches are accepted directly.
    Non-exact pairs are retained when:
    - SimHash Hamming distance is below the adaptive threshold, and
    - TF-IDF lexical similarity is above the recall-oriented threshold.
    """
    exact_matches = []
    candidate_pairs = []

    for suspicious_text in suspicious_sentences:
        suspicious_hash = compute_simhash(suspicious_text, SIMHASH_NGRAM_SIZE)
        match_found = False

        for source_text in source_sentences:
            source_hash = compute_simhash(source_text, SIMHASH_NGRAM_SIZE)
            distance = hamming_distance(suspicious_hash, source_hash)

            if distance == 0:
                exact_matches.append((suspicious_text, 1.0))
                match_found = True
                break

        if match_found:
            continue

        suspicious_lemma = lemmatize_text(suspicious_text, nlp)

        for source_text in source_sentences:
            source_lemma = lemmatize_text(source_text, nlp)

            distance = hamming_distance(
                compute_simhash(suspicious_lemma, SIMHASH_NGRAM_SIZE),
                compute_simhash(source_lemma, SIMHASH_NGRAM_SIZE)
            )

            adaptive_threshold = (
                SHORT_SENTENCE_K
                if len(suspicious_lemma.split()) < SHORT_SENTENCE_TOKEN_LIMIT
                else LONG_SENTENCE_K
            )

            if distance == 0:
                exact_matches.append((suspicious_text, 1.0))
                match_found = True
                break

            if (
                distance < adaptive_threshold
                and lexical_similarity(suspicious_lemma, source_lemma)
                > LEXICAL_SIMILARITY_THRESHOLD
            ):
                candidate_pairs.append((suspicious_text, source_text, distance))

    return exact_matches, candidate_pairs


def load_verifier(verifier_name: str, device):
    """
    Load the selected semantic verifier.
    """
    cfg = VERIFIER_CONFIG[verifier_name]

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_path"],
        trust_remote_code=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_path"],
        trust_remote_code=True,
        use_flash_attn=False
    )

    if torch.cuda.is_available():
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device, dtype=torch.float32)

    model.eval()

    return tokenizer, model, cfg


def classify_candidate_pairs(
    pairs,
    tokenizer,
    model,
    cfg,
    device
):
    """
    Apply the selected semantic verifier to candidate sentence pairs.
    """
    if not pairs:
        return []

    accepted = []
    batch_size = cfg["batch_size"]
    threshold = cfg["threshold"]
    max_length = cfg["max_length"]

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
            logits = model(**inputs).logits.view(-1)
            probabilities = torch.sigmoid(logits).detach().float().cpu().numpy()

        for j, score in enumerate(probabilities):
            if score >= threshold:
                accepted.append((batch[j][0], float(score)))

        del inputs, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return accepted


def lemmatize_tokens(text: str, nlp) -> list[str]:
    """
    Convert text to lemmatised tokens.
    """
    return lemmatize_text(text, nlp).split()


def compute_token_metrics(gold_tokens: list[str], predicted_tokens: list[str]):
    """
    Compute set-based token precision, recall, and F1.
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


def evaluate(input_csv: str, output_dir: str, verifier_name: str):
    """
    Run long-document semantic reuse evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Verifier: {verifier_name}")

    nlp = stanza.Pipeline(
        "kk",
        processors="tokenize,lemma",
        use_gpu=torch.cuda.is_available()
    )

    tokenizer, model, cfg = load_verifier(verifier_name, device)

    df = pd.read_csv(input_csv)
    results = []

    start_time = time.time()

    for idx, row in df.iterrows():
        print(
            f"Processing {idx + 1}/{len(df)}: "
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
            nlp
        )

        verifier_matches = classify_candidate_pairs(
            candidate_pairs,
            tokenizer,
            model,
            cfg,
            device
        )

        predicted_text = " ".join(
            text for text, _ in simhash_matches + verifier_matches
        )

        predicted_tokens = lemmatize_tokens(predicted_text, nlp)
        gold_tokens = lemmatize_tokens(str(row["suspicious_words"]), nlp)

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

    results_df = pd.DataFrame(results)

    metrics_path = os.path.join(
        output_dir,
        f"{verifier_name}_evaluation_metrics_{timestamp}.csv"
    )

    results_df.to_csv(metrics_path, index=False)

    print("Saved evaluation metrics to:", metrics_path)
    print("Average metrics:")
    print(f"Precision: {results_df['precision'].mean():.4f}")
    print(f"Recall:    {results_df['recall'].mean():.4f}")
    print(f"F1 Score:  {results_df['f1'].mean():.4f}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

    del df, results, results_df
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate long-document semantic reuse detection."
    )

    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to input CSV containing suspicious/source document pairs."
    )

    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory for output metric files."
    )

    parser.add_argument(
        "--verifier",
        default="jina_v2_base_multi",
        choices=list(VERIFIER_CONFIG.keys()),
        help="Semantic verifier to use."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        verifier_name=args.verifier
    )
