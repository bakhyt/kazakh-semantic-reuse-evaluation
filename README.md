# Kazakh Semantic Reuse Evaluation

This repository contains code, configurations, results, and documentation for the CLEF 2026 proceedings paper:

**Model Comparison for Kazakh Semantic Reuse Detection Across Pairwise, News, and Long-Document Evaluation**

Authors: **Bakhyt Bakiyev, Shuo Wang, and Mubashir Ali**

## Overview

This project evaluates models for Kazakh semantic reuse detection across three settings:

1. Pairwise PAN-KK classification
2. External Kazakh news-domain snippet-pair classification
3. Long-document semantic reuse detection using a three-stage pipeline

The main goal is to show that model rankings change across evaluation setting, domain, and document scale. Therefore, pairwise benchmark performance alone is not sufficient for selecting models for practical long-document Kazakh semantic reuse detection.

## Evaluation Settings

### 1. Pairwise PAN-KK Evaluation

The pairwise evaluation uses the PAN-KK Kazakh PAN-style benchmark introduced in the authors' previous work. Models are evaluated on suspicious-source text pairs using precision, recall, and F1.

The full PAN-KK dataset is not redistributed in this repository.

### 2. External News-Domain Evaluation

The news-domain evaluation uses a controlled benchmark constructed from Nur.kz and Tengrinews.kz. The benchmark contains 1,000 snippet pairs, with 500 similar and 500 non-similar pairs.

The news benchmark is balanced and selection-biased. It is intended for controlled external-domain semantic discrimination, not for estimating the natural prevalence of cross-portal reuse.

Full news articles are not redistributed because copyright remains with the original publishers. Where possible, this repository provides metadata, labels, URLs or identifiers, annotation documentation, and reconstruction instructions.

### 3. Long-Document Evaluation

The long-document evaluation uses a three-stage framework:

1. Preprocessing
2. Candidate retrieval
3. Detailed analysis

The system processes suspicious and source documents, retrieves candidate sentence pairs, applies a semantic decision model, and compares predicted reused text against gold reused text.

## Relation to Previous PAN-KK Repository

The PAN-KK dataset and original Kazakh extrinsic plagiarism detection pipeline were introduced in the previous project repository:

https://github.com/bakhyt/kazakh-extrinsic-plagiarism-detection

The present repository extends that work by focusing on model comparison across three evaluation settings:

- pairwise PAN-KK classification
- external Kazakh news-domain snippet-pair evaluation
- long-document semantic reuse detection

The full datasets are not duplicated in this repository. This repository provides code templates, configuration files, documentation, result tables, and synthetic sample input formats.

## Repository Structure

```text
configs/                   Model and threshold configuration files
preprocessing/             Text cleaning, sentence segmentation, tokenisation, and lemmatisation
candidate_retrieval/       SimHash and TF-IDF candidate retrieval
pairwise_evaluation/       Scripts for PAN-KK pairwise model evaluation
news_evaluation/           Scripts for news-domain snippet-pair evaluation
long_document_evaluation/  Scripts for long-document pipeline evaluation
retrieval_ablation/        Retrieval-only ablation experiments
results/                   Tables and result files used in the paper
data/                      Dataset documentation and synthetic sample input formats
docs/                      Additional documentation
```
