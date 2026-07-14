"""
Template for long-document semantic reuse evaluation.

This script shows the evaluation structure used in the CLEF 2026 paper:
1. preprocessing
2. candidate retrieval
3. semantic verification
4. token-level evaluation

Private paths, trained model locations, and non-redistributable data are not included.
Users should provide their own model path and input CSV.
"""

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
# Configuration
# =========================================================

DEFAULT_CONFIG = {
    "verifier": "jina_v2_base_multi",
    "model_type": "jina_v2_classifier",
    "threshold": 0.60,
    "batch_size": 8,
    "max_length": 1024,
    "simhash_n": 1,
    "short_sentence_k": 24,
    "long_sentence_k": 31,
    "short_sentence_condition": "lemma_token_count < 10",
    "min_sentence_token_length": 2,
    "lexical_similarity_threshold": 0.10,
}


# =========================================================
# Preprocessing
# =========================================================

def preprocess_kazakh_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s.!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str, nlp, min_token_len: int):
    doc = nlp(preprocess_kazakh_text(text))
    return [
        sentence.text
        for sentence in doc.sentences
        if len(sentence.words) >= min_token_len
    ]


def lemmatize_text(text: str, nlp, cache: dict):
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
# Candidate Retrieval
# =========================================================

def get_shingles(text: str, n: int):
    words = text.split()
    if len(words) < n:
        return [text]
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def compute_simhash(text: str, n: int, cache: dict):
    if text in cache:
        return cache[text]

    value = Simhash(get_shingles(text, n)).value
    cache[text] = value
    return value


def lexical_similarity(text1: str, text2: str) -> float:
    vectorizer = TfidfVectorizer().fit([text1, text2])
    vectors = vectorizer.transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]


def generate_candidate_pairs(
    suspicious_sentences,
    source_sentences,
    nlp,
    lemmatize_cache,
    simhash_cache,
    config
):
    exact_matches = []
    candidate_pairs = []

    for suspicious_sentence in suspicious_sentences:
        suspicious_hash = compute_simhash(
            suspicious_sentence,
            config["simhash_n"],
            simhash_cache
        )

        match_found = False

        # Exact SimHash match on original sentence text
        for source_sentence in source_sentences:
            source_hash = compute_simhash(
                source_sentence,
                config["simhash_n"],
                simhash_cache
            )

            distance = bin(suspicious_hash ^ source_hash).count("1")

            if distance == 0:
                exact_matches.append((suspicious_sentence, 1.0))
                match_found = True
                break

        if match_found:
            continue

        suspicious_lemma = lemmatize_text(
            suspicious_sentence,
            nlp,
            lemmatize_cache
        )

        for source_sentence in source_sentences:
            source_lemma = lemmatize_text(
                source_sentence,
                nlp,
                lemmatize_cache
            )

            suspicious_lemma_hash = compute_simhash(
                suspicious_lemma,
                config["simhash_n"],
                simhash_cache
            )
            source_lemma_hash = compute_simhash(
                source_lemma,
                config["simhash_n"],
                simhash_cache
            )

            distance = bin(suspicious_lemma_hash ^ source_lemma_hash).count("1")

            adaptive_k = (
                config["short_sentence_k"]
                if len(suspicious_lemma.split()) < 10
                else config["long_sentence_k"]
            )

            if distance == 0:
                exact_matches.append((suspicious_sentence, 1.0))
                match_found = True
                break

            if (
                distance < adaptive_k
                and lexical_similarity(suspicious_lemma, source_lemma)
                > config["lexical_similarity_threshold"]
            ):
                candidate_pairs.append(
                    (suspicious_sentence, source_sentence, distance)
                )

    return exact_matches, candidate_pairs


# =========================================================
# Semantic Verification
# =========================================================

def load_verifier(model_path: str, config: dict, device):
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
    return tokenizer, model


def classify_candidate_pairs(pairs, tokenizer, model, config, device):
    if not pairs:
        return []

    accepted = []
    batch_size = config["batch_size"]
    threshold = config["threshold"]
    max_length = config["max_length"]

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]

        suspicious_batch = [suspicious for suspicious, source, _ in batch]
        source_batch = [source for suspicious, source, _ in batch]

        inputs = tokenizer(
            suspicious_batch,
            text_pair=source_batch,
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


# =========================================================
# Evaluation Metrics
# =========================================================

def lemmatize_tokens(text: str, nlp, cache: dict):
    return lemmatize_text(text, nlp, cache).split()


def compute_token_metrics(gold_tokens, predicted_tokens):
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
# Main Evaluation
# =========================================================

def run_evaluation(input_csv: str, model_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    config = DEFAULT_CONFIG
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nlp = stanza.Pipeline(
        "kk",
        processors="tokenize,lemma",
        use_gpu=torch.cuda.is_available()
    )

    tokenizer, model = load_verifier(model_path, config, device)

    lemmatize_cache = {}
    simhash_cache = {}

    df = pd.read_csv(input_csv)

    results = []
    start_time = time.time()

    required_columns = [
        "suspicious_document",
        "source_document",
        "suspicious_content",
        "source_content",
        "suspicious_words"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    for index, row in df.iterrows():
        if pd.isna(row["suspicious_content"]) or pd.isna(row["source_content"]):
            continue

        suspicious_sentences = split_into_sentences(
            str(row["suspicious_content"]),
            nlp,
            config["min_sentence_token_length"]
        )

        source_sentences = split_into_sentences(
            str(row["source_content"]),
            nlp,
            config["min_sentence_token_length"]
        )

        simhash_matches, candidate_pairs = generate_candidate_pairs(
            suspicious_sentences,
            source_sentences,
            nlp,
            lemmatize_cache,
            simhash_cache,
            config
        )

        verifier_matches = classify_candidate_pairs(
            candidate_pairs,
            tokenizer,
            model,
            config,
            device
        )

        predicted_text = " ".join(
            text for text, _ in simhash_matches + verifier_matches
        )

        predicted_tokens = lemmatize_tokens(
            predicted_text,
            nlp,
            lemmatize_cache
        )

        gold_tokens = lemmatize_tokens(
            str(row["suspicious_words"]),
            nlp,
            lemmatize_cache
        )

        precision, recall, f1 = compute_token_metrics(
            gold_tokens,
            predicted_tokens
        )

        results.append({
            "verifier": config["verifier"],
            "suspicious_document": row["suspicious_document"],
            "source_document": row["source_document"],
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    result_df = pd.DataFrame(results)

    output_path = os.path.join(
        output_dir,
        f"{config['verifier']}_long_document_metrics_{timestamp}.csv"
    )

    result_df.to_csv(output_path, index=False)

    print("Saved evaluation metrics to:", output_path)
    print("Average precision:", round(result_df["precision"].mean(), 4))
    print("Average recall:", round(result_df["recall"].mean(), 4))
    print("Average F1:", round(result_df["f1"].mean(), 4))
    print("Runtime seconds:", round(time.time() - start_time, 2))

    del df, result_df, results
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Template for long-document semantic reuse evaluation."
    )

    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to input CSV containing suspicious/source document pairs."
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path or Hugging Face identifier of the verifier model."
    )

    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory where evaluation results will be saved."
    )

    args = parser.parse_args()

    run_evaluation(
        input_csv=args.input_csv,
        model_path=args.model_path,
        output_dir=args.output_dir
    )
