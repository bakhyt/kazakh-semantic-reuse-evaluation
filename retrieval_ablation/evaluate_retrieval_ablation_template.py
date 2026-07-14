"""
Retrieval ablation evaluation template.

This script evaluates candidate retrieval variants for long-document
semantic reuse detection.

Supported modes:
- simhash_tfidf
- simhash_only
- tfidf_only
- tfidf_char35_only

Expected input CSV columns:
- suspicious_document
- source_document
- suspicious_content
- source_content
- suspicious_words

The script does not save gold text, predicted text, or error-analysis text
to avoid redistributing dataset content.
"""

import argparse
import datetime
import gc
import os
import re
import time

import numpy as np
import pandas as pd
import stanza
import torch
from simhash import Simhash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SIMHASH_NGRAM_SIZE = 1
SHORT_SENTENCE_K = 24
LONG_SENTENCE_K = 31
SHORT_SENTENCE_TOKEN_LIMIT = 10
MIN_SENTENCE_TOKEN_LENGTH = 2

LEXICAL_SIMILARITY_THRESHOLD = 0.10
CHAR35_SIMILARITY_THRESHOLD = 0.45

lemmatize_text_cache = {}
simhash_cache = {}
lexical_cache = {}
char35_cache = {}


def preprocess_kazakh_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s.!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str, nlp) -> list[str]:
    doc = nlp(preprocess_kazakh_text(text))
    return [
        sentence.text
        for sentence in doc.sentences
        if len(sentence.words) >= MIN_SENTENCE_TOKEN_LENGTH
    ]


def lemmatize_text(text: str, nlp) -> str:
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


def lemmatize_tokens(text: str, nlp) -> list[str]:
    return lemmatize_text(text, nlp).split()


def get_shingles(text: str, n: int) -> list[str]:
    words = text.split()

    if len(words) < n:
        return [text]

    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def compute_simhash(text: str, n: int) -> int:
    key = (text, n)

    if key in simhash_cache:
        return simhash_cache[key]

    value = Simhash(get_shingles(text, n)).value
    simhash_cache[key] = value
    return value


def hamming_distance(hash_a: int, hash_b: int) -> int:
    return bin(hash_a ^ hash_b).count("1")


def lexical_similarity(text1: str, text2: str) -> float:
    key = (text1, text2)

    if key in lexical_cache:
        return lexical_cache[key]

    try:
        vectorizer = TfidfVectorizer().fit([text1, text2])
        vectors = vectorizer.transform([text1, text2])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    except Exception:
        similarity = 0.0

    lexical_cache[key] = float(similarity)
    return float(similarity)


def lexical_similarity_char35(text1: str, text2: str) -> float:
    key = (text1, text2)

    if key in char35_cache:
        return char35_cache[key]

    try:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5)
        ).fit([text1, text2])

        vectors = vectorizer.transform([text1, text2])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    except Exception:
        similarity = 0.0

    char35_cache[key] = float(similarity)
    return float(similarity)


def run_simhash_tfidf(suspicious_sentences, source_sentences, nlp):
    predicted = []

    for suspicious_text in suspicious_sentences:
        suspicious_hash = compute_simhash(
            suspicious_text,
            SIMHASH_NGRAM_SIZE
        )

        match_found = False

        for source_text in source_sentences:
            source_hash = compute_simhash(source_text, SIMHASH_NGRAM_SIZE)
            distance = hamming_distance(suspicious_hash, source_hash)

            if distance == 0:
                predicted.append(suspicious_text)
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
                predicted.append(suspicious_text)
                match_found = True
                break

            if (
                distance < adaptive_threshold
                and lexical_similarity(suspicious_lemma, source_lemma)
                > LEXICAL_SIMILARITY_THRESHOLD
            ):
                predicted.append(suspicious_text)
                match_found = True
                break

    return predicted


def run_simhash_only(suspicious_sentences, source_sentences, nlp):
    predicted = []

    for suspicious_text in suspicious_sentences:
        suspicious_lemma = lemmatize_text(suspicious_text, nlp)
        suspicious_hash = compute_simhash(
            suspicious_lemma,
            SIMHASH_NGRAM_SIZE
        )

        for source_text in source_sentences:
            source_lemma = lemmatize_text(source_text, nlp)
            source_hash = compute_simhash(source_lemma, SIMHASH_NGRAM_SIZE)

            distance = hamming_distance(suspicious_hash, source_hash)

            adaptive_threshold = (
                SHORT_SENTENCE_K
                if len(suspicious_lemma.split()) < SHORT_SENTENCE_TOKEN_LIMIT
                else LONG_SENTENCE_K
            )

            if distance == 0 or distance < adaptive_threshold:
                predicted.append(suspicious_text)
                break

    return predicted


def run_tfidf_only(suspicious_sentences, source_sentences, nlp):
    predicted = []

    for suspicious_text in suspicious_sentences:
        suspicious_lemma = lemmatize_text(suspicious_text, nlp)

        for source_text in source_sentences:
            source_lemma = lemmatize_text(source_text, nlp)
            similarity = lexical_similarity(suspicious_lemma, source_lemma)

            if similarity > LEXICAL_SIMILARITY_THRESHOLD:
                predicted.append(suspicious_text)
                break

    return predicted


def run_tfidf_char35_only(suspicious_sentences, source_sentences):
    predicted = []

    for suspicious_text in suspicious_sentences:
        for source_text in source_sentences:
            similarity = lexical_similarity_char35(
                suspicious_text,
                source_text
            )

            if similarity > CHAR35_SIMILARITY_THRESHOLD:
                predicted.append(suspicious_text)
                break

    return predicted


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


def evaluate_retrieval_ablation(
    input_csv: str,
    output_dir: str,
    mode: str,
    max_pairs: int | None = None
):
    os.makedirs(output_dir, exist_ok=True)

    nlp = stanza.Pipeline(
        "kk",
        processors="tokenize,lemma",
        use_gpu=torch.cuda.is_available()
    )

    df = pd.read_csv(input_csv)

    if max_pairs is not None:
        df = df.head(max_pairs)

    results = []
    start_time = time.time()

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

        if mode == "simhash_tfidf":
            predicted_sentences = run_simhash_tfidf(
                suspicious_sentences,
                source_sentences,
                nlp
            )

        elif mode == "simhash_only":
            predicted_sentences = run_simhash_only(
                suspicious_sentences,
                source_sentences,
                nlp
            )

        elif mode == "tfidf_only":
            predicted_sentences = run_tfidf_only(
                suspicious_sentences,
                source_sentences,
                nlp
            )

        elif mode == "tfidf_char35_only":
            predicted_sentences = run_tfidf_char35_only(
                suspicious_sentences,
                source_sentences
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        predicted_text = " ".join(predicted_sentences)

        predicted_tokens = lemmatize_tokens(predicted_text, nlp)
        gold_tokens = lemmatize_tokens(str(row["suspicious_words"]), nlp)

        precision, recall, f1 = compute_token_metrics(
            gold_tokens,
            predicted_tokens
        )

        results.append({
            "mode": mode,
            "suspicious_document": row["suspicious_document"],
            "source_document": row["source_document"],
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    results_df = pd.DataFrame(results)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    output_path = os.path.join(
        output_dir,
        f"{mode}_ablation_metrics_{timestamp}.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("Saved evaluation metrics to:", output_path)
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
        description="Evaluate retrieval ablation methods."
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
        "--mode",
        required=True,
        choices=[
            "simhash_tfidf",
            "simhash_only",
            "tfidf_only",
            "tfidf_char35_only"
        ],
        help="Retrieval ablation mode."
    )

    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Optional maximum number of document pairs to evaluate."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate_retrieval_ablation(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        mode=args.mode,
        max_pairs=args.max_pairs
    )
