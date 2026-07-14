# Candidate Retrieval Description

Candidate retrieval is Stage 2 of the long-document evaluation pipeline.

The purpose of candidate retrieval is to reduce the number of suspicious-source sentence pairs before applying more expensive semantic verification models.

## Methods

The candidate retrieval stage uses:

- SimHash filtering
- TF-IDF cosine similarity filtering

## SimHash

SimHash is used to produce compact fingerprints for candidate sentence comparison.

The current configuration file is:

```text
configs/simhash_thresholds.json
```

The final camera-ready version should specify:

- whether SimHash uses token-based shingles or character n-grams
- the fingerprint size
- the adaptive Hamming-distance thresholds
- whether the thresholds were tuned or selected empirically

## TF-IDF

TF-IDF cosine similarity is used as a recall-oriented lexical filter.

The long-document candidate retrieval threshold is:

```text
0.10
```

This threshold is used to avoid prematurely removing potentially reused or paraphrased sentence pairs before semantic verification.
