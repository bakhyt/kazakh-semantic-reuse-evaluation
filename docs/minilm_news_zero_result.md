# MiniLM Zero Result in News-Domain Evaluation

In the external news-domain snippet-pair evaluation, MiniLM obtained:

- Accuracy: 0.4988
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000

This happened because MiniLM predicted no positive examples in the held-out news-domain test set at the selected threshold.

Therefore:

- true positives = 0
- false positives = 0
- recall = 0 because no positive gold examples were recovered
- precision and F1 are reported as 0.0000 under the zero-division convention

The accuracy of 0.4988 reflects the approximately balanced test set and does not indicate useful discrimination.
