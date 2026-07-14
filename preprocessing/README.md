# Preprocessing

This folder contains scripts and documentation for Kazakh text preprocessing.

## Files

- `preprocess_text.py`  
  Provides basic text normalisation, sentence segmentation, tokenisation, and lemmatisation.

- `preprocessing_description.md`  
  Describes the preprocessing steps used before candidate retrieval and model evaluation.

## Preprocessing Steps

The preprocessing stage includes:

- lowercasing
- digit removal
- punctuation normalisation
- repeated whitespace removal
- sentence segmentation
- tokenisation
- lemmatisation

Sentence segmentation, tokenisation, and lemmatisation are performed using the Kazakh pipeline in Stanza.

## Example Usage

```bash
python preprocess_text.py
```

## Note

The preprocessing script is provided as a public template. It does not include private dataset files or local paths.
