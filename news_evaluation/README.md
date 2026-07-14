# News-Domain Evaluation

This folder contains scripts for evaluating semantic reuse detection on the external Kazakh news-domain benchmark.

## Files

- `evaluate_news_predictions.py`  
  Converts ordinal labels to binary labels and computes accuracy, precision, recall, and F1.

## Label Mapping

The original news annotation uses a 4-point ordinal scale:

- 0 = dissimilar
- 1 = related or partially similar, but no clear semantic reuse
- 2 = clear semantic overlap or reuse
- 3 = near-identical or identical meaning

For binary evaluation:

- labels 0--1 are mapped to non-similar
- labels 2--3 are mapped to similar

## Expected Input Format

The evaluation script expects a CSV file with the following columns:

```text
gold_label
predicted_label
```

## Example Usage

```bash
python evaluate_news_predictions.py news_predictions.csv
```

python evaluate_news_models_template.py \
  --input_csv path/to/news_pipeline_vs_gold.csv \
  --output_csv outputs/news_model_results.csv

python evaluate_news_additional_models_template.py \
  --input_csv path/to/news_pipeline_vs_gold.csv \
  --distilbert_model_path path/to/distilbert-model \
  --minilm_model_path path/to/minilm-model \
  --bertmulti_model_path path/to/bertmulti-model \
  --sbert_mpnet_model_path path/to/sbert-mpnet-model \
  --output_csv outputs/news_additional_model_results.csv
