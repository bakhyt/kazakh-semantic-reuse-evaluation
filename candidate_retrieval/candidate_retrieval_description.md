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
