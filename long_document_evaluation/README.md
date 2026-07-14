# Long-Document Evaluation

This folder contains scripts and documentation for the long-document semantic reuse detection pipeline.

## Files

- `long_document_protocol.md`  
  Describes the long-document evaluation protocol.

- `evaluate_token_level.py`  
  Computes token-level precision, recall, and F1 from gold reused text and predicted reused text.

- `evaluate_long_document_pipeline_template.py`  
  Public template version of the long-document evaluation pipeline.

## Public Script Note

The public long-document pipeline script removes local paths and does not save full gold text, predicted text, or detailed error-analysis text. This avoids redistributing dataset text while preserving the evaluation logic and reported metrics.

## Expected Input Format

The long-document evaluation scripts expect a CSV file with columns similar to:

```text
suspicious_document
source_document
suspicious_content
source_content
suspicious_words
```

A synthetic example is provided in:

```text
data/sample_input_formats/long_document_sample.csv
```

## Example Usage

```bash
python evaluate_long_document_pipeline_template.py \
  --input_csv path/to/matched_suspicious_source_with_words.csv \
  --output_dir outputs \
  --verifier jina_v2_base_multi
```

For token-level evaluation only:

```bash
python evaluate_token_level.py long_document_predictions.csv
```

## Reported Results

The reported long-document results are stored in:

```text
results/long_document_results.csv
```

## Note

The full long-document dataset is not redistributed in this repository.
