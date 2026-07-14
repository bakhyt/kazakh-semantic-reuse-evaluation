# Reviewer Response Notes

This document summarises how the repository addresses the main reviewer comments for the CLEF 2026 camera-ready paper.

## Terminology

The paper uses **semantic reuse detection** for model outputs because the system detects textual or semantic overlap. It does not determine intent, attribution, or academic misconduct.

## Preprocessing

The repository documents preprocessing in:

```text
preprocessing/preprocessing_description.md
```

Preprocessing includes normalisation, sentence segmentation, tokenisation, and lemmatisation using Stanza.

## Candidate Retrieval

Candidate retrieval is documented in:

```text
candidate_retrieval/candidate_retrieval_description.md
configs/simhash_thresholds.json
```

This includes SimHash, TF-IDF, the 64-bit fingerprint setting, the TF-IDF threshold of 0.10, and the adaptive SimHash threshold configuration.

## MiniLM News-Domain Result

The MiniLM zero result is documented in:

```text
docs/minilm_news_zero_result.md
```

MiniLM predicted no positive examples in the held-out news-domain test set at the selected threshold. Therefore, precision, recall, and F1 are reported as 0.0000 under the zero-division convention.

## News Benchmark

The news benchmark is documented in:

```text
data/news_benchmark/benchmark_description.md
```

The benchmark is balanced and selection-biased. It is intended for controlled external-domain semantic discrimination, not prevalence estimation.

## Long-Document Evaluation

The long-document protocol is documented in:

```text
long_document_evaluation/long_document_protocol.md
```

The evaluation uses the full three-stage pipeline and reports token-level precision, recall, F1, and runtime.

## Data Availability

PAN-KK is available through Zenodo:

https://doi.org/10.5281/zenodo.17538305

Full news articles are not redistributed because copyright remains with the original publishers.

## Remaining Items to Confirm

The following items still need to be checked against the final implementation before the camera-ready submission:

- exact SimHash feature type: token-based shingles or character n-grams
- exact adaptive SimHash Hamming-distance thresholds
- exact reason for selecting the TF-IDF threshold of 0.10
- exact MiniLM threshold and prediction counts for the news-domain test set
- final code and script names for reproducing each result table
