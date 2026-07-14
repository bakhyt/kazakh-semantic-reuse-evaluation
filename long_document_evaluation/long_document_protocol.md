# Long-Document Evaluation Protocol

The long-document evaluation tests the full three-stage pipeline on PAN-KK long-document pairs.

## Pipeline

The evaluation uses the following stages:

1. Preprocessing
2. Candidate retrieval
3. Detailed analysis

## Evaluation Unit

The evaluation unit is a suspicious-source document pair.

Documents are segmented into sentences. Candidate suspicious-source sentence pairs are generated during candidate retrieval. The selected semantic model then scores candidate pairs.

Accepted suspicious-side sentences are combined to form the predicted reused text for each suspicious document.

## Metrics

Predicted reused text is compared against gold reused text.

The main reported metrics are:

- token-level precision
- token-level recall
- token-level F1
- runtime for the final 1,000-pair comparison

## Subset Sizes

The experiments were conducted in three stages:

- 50 document pairs for initial screening
- 100 document pairs for intermediate comparison
- 1,000 document pairs for final comparison

## Important Note

Long-document evaluation is different from pairwise classification and news-domain snippet-pair classification. It evaluates the interaction between preprocessing, candidate retrieval, and semantic verification at document scale.
