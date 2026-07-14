## Example Usage

### General prediction evaluation

```bash
python evaluate_pairwise_predictions.py predictions.csv
```

The input CSV should contain:

```text
gold_label
predicted_label
```

### Jina v2 pairwise evaluation

```bash
python evaluate_pairwise_jina_v2_template.py \
  --model_path path/to/jina-v2-model \
  --test_csv path/to/test-balanced-4562.csv \
  --threshold 0.60
```

### TF-IDF char 3-5 baseline

```bash
python evaluate_pairwise_tfidf_char35_template.py \
  --test_csv path/to/test-balanced-4562.csv \
  --threshold 0.45
```
