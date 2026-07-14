# Preprocessing Description

Preprocessing is used before candidate retrieval and model evaluation.

The preprocessing stage includes:

- lowercasing
- digit removal
- punctuation normalisation
- repeated whitespace removal
- sentence segmentation
- tokenisation
- lemmatisation

Sentence segmentation, tokenisation, and lemmatisation are performed using the Kazakh pipeline in Stanza.

This preprocessing is especially important for Kazakh because Kazakh is morphologically rich and agglutinative, so semantically related words may appear in many surface forms.
