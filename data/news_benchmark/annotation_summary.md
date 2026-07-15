# News Benchmark Annotation Summary

The external Kazakh news-domain benchmark was manually evaluated by native Kazakh speakers.

## Annotation Scale

Annotators used a four-point ordinal similarity scale:

- 0 = dissimilar
- 1 = related or partially similar, but no clear semantic reuse
- 2 = clear semantic overlap or reuse
- 3 = near-identical or substantially equivalent meaning

For binary evaluation:

- labels 0--1 were mapped to non-similar
- labels 2--3 were mapped to similar

## Annotation Design

The benchmark contains 1,000 snippet pairs:

- 500 similar pairs
- 500 non-similar pairs

The benchmark is balanced and controlled. It is designed for model comparison, not for estimating the natural frequency of reuse between news portals.

## Annotator Agreement

A subset of the benchmark was annotated by multiple annotators to assess reliability.

| Subset | Annotators | Metric | Score |
|---|---:|---|---:|
| 20 fully shared pairs | 6 | Krippendorff's alpha | 0.990 |
| 97-pair overlap block | 2 | Exact agreement | 0.948 |
| 97-pair overlap block | 2 | Quadratic weighted kappa | 0.981 |

These agreement scores indicate strong consistency among Kazakh-speaking annotators.

## Important Limitation

Most benchmark pairs were not annotated by all annotators. Therefore, the agreement scores describe the overlap subsets, not the entire 1,000-pair benchmark.

Full news articles are not redistributed in this repository because copyright remains with the original publishers.
