python evaluate_pairwise_jina_v2_template.py \
  --model_path path/to/jina-v2-model \
  --test_csv path/to/test-balanced-4562.csv \
  --threshold 0.60

python evaluate_pairwise_tfidf_char35_template.py \
  --test_csv path/to/test-balanced-4562.csv \
  --threshold 0.45
