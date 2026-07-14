# Retrieval Ablation

This folder contains scripts for retrieval-only ablation experiments in the long-document evaluation setting.

## Modes

The ablation script supports four modes:

- `simhash_tfidf`
- `simhash_only`
- `tfidf_only`
- `tfidf_char35_only`

## Example Usage

```bash
python evaluate_retrieval_ablation_template.py \
  --input_csv path/to/matched_suspicious_source_with_words.csv \
  --output_dir outputs \
  --mode tfidf_char35_only \
  --max_pairs 50
```

## Reported Results

The reported retrieval ablation results are stored in:

```text
results/retrieval_ablation_results.csv
```
