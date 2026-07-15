"""
SimHash candidate retrieval template.

This script provides a public template for SimHash-based candidate retrieval
used in the long-document semantic reuse detection pipeline.

It does not include private dataset paths or full dataset files.
"""

from simhash import Simhash


def get_token_shingles(text: str, n: int = 1):
    """
    Create token-based n-gram shingles.

    In the reported experiments, n=1 corresponds to token-unigram SimHash.
    """
    tokens = str(text).split()

    if len(tokens) < n:
        return [str(text)]

    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_simhash(text: str, n: int = 1) -> int:
    """
    Compute a SimHash fingerprint from token shingles.
    """
    return Simhash(get_token_shingles(text, n)).value


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """
    Compute Hamming distance between two SimHash fingerprints.
    """
    return bin(hash_a ^ hash_b).count("1")


def retrieve_simhash_candidates(
    suspicious_sentences,
    source_sentences,
    short_sentence_k: int = 24,
    long_sentence_k: int = 31,
    short_sentence_token_limit: int = 10,
    ngram_size: int = 1
):
    """
    Retrieve candidate suspicious-source sentence pairs using adaptive SimHash thresholds.

    Short suspicious sentences use short_sentence_k.
    Longer suspicious sentences use long_sentence_k.
    """
    candidates = []

    for suspicious_index, suspicious_text in enumerate(suspicious_sentences):
        suspicious_hash = compute_simhash(suspicious_text, ngram_size)
        suspicious_length = len(str(suspicious_text).split())

        adaptive_k = (
            short_sentence_k
            if suspicious_length < short_sentence_token_limit
            else long_sentence_k
        )

        for source_index, source_text in enumerate(source_sentences):
            source_hash = compute_simhash(source_text, ngram_size)
            distance = hamming_distance(suspicious_hash, source_hash)

            if distance == 0 or distance < adaptive_k:
                candidates.append({
                    "suspicious_index": suspicious_index,
                    "source_index": source_index,
                    "hamming_distance": distance
                })

    return candidates


if __name__ == "__main__":
    suspicious = [
        "Қазақстанда жасанды интеллект зерттеулері дамып келеді.",
        "Бұл сөйлем басқа тақырып туралы."
    ]

    source = [
        "Қазақстанда AI зерттеулері дамуда.",
        "Ауа райы бүгін жылы болады."
    ]

    candidates = retrieve_simhash_candidates(suspicious, source)
    print(candidates)
