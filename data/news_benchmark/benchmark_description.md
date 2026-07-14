# External Kazakh News Benchmark

The external Kazakh news benchmark was constructed from Kazakh news articles collected from Nur.kz and Tengrinews.kz.

## Benchmark Size

The benchmark contains 1,000 snippet pairs:

- 500 similar pairs
- 500 non-similar pairs

## Annotation

The benchmark was manually annotated by native Kazakh speakers using a 4-point ordinal scale:

- 0 = dissimilar
- 1 = related or partially similar, but no clear semantic reuse
- 2 = clear semantic overlap or reuse
- 3 = near-identical or identical meaning

For binary evaluation:

- labels 0--1 are mapped to non-similar
- labels 2--3 are mapped to similar

## Important Limitation

The benchmark is balanced and selection-biased because pairs were sampled from preliminary predicted similar and non-similar candidates.

Therefore, it should be interpreted as a controlled external-domain semantic discrimination benchmark, not as an estimate of the natural prevalence of cross-portal reuse.

## Copyright Notice

Full news articles are not redistributed because copyright remains with the original publishers.

Where possible, this repository provides only legally shareable materials such as labels, metadata, URLs or identifiers, split IDs, and reconstruction instructions.
