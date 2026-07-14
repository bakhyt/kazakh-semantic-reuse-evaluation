# Pairwise Evaluation

This folder contains scripts for evaluating pairwise semantic reuse classification on PAN-KK.

## Files

- `evaluate_pairwise_predictions.py`  
  Computes accuracy, precision, recall, and F1 from gold and predicted binary labels.

## Expected Input Format

The evaluation script expects a CSV file with the following columns:

```text
gold_label
predicted_label
