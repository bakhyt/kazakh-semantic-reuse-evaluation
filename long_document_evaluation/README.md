# Long-Document Evaluation

This folder contains documentation and scripts for evaluating the long-document semantic reuse detection pipeline.

## Files

- `long_document_protocol.md`  
  Explains the long-document evaluation protocol.

- `evaluate_token_level.py`  
  Computes token-level precision, recall, and F1 from gold reused text and predicted reused text.

## Expected Input Format

The evaluation script expects a CSV file with the following columns:

```text
gold_reused_text
predicted_reused_text
