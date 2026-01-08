python -m experiments.bscore --context continuous --dataset "ilenia|language:english|annotated:no" --export-csv "runs-ilenia/en/aggregation.csv" --use-root yes --lower-proper-nouns yes --semantic-check no --split-endings no --inspect-professions no --use-morph no >> "runs-ilenia/en/.log.txt"

#

python -m experiments.bscore.zfilter --path runs-ilenia/en/aggregation.csv --zscore-threshold 2

# 

python -m experiments.bscore.filterby --path "runs-ilenia/en/aggregation-z2filtered.csv" --pos ADJ

# 

python -m experiments.bscore.filterby --path "runs-ilenia/en/aggregation-z2filtered.csv" --pos VERB

# 

python -m experiments.bscore.disparityfilter --path runs-ilenia/en/aggregation.csv

# 

python -m experiments.bscore.filterby --path "runs-ilenia/en/aggregation-disparity-filtered-(0.5).csv" --pos ADJ

# 

python -m experiments.bscore.filterby --path "runs-ilenia/en/aggregation-disparity-filtered-(0.5).csv" --pos VERB

# 

python -m experiments.bscore.getstats --path "runs-ilenia/en/aggregation-disparity-filtered-(0.5).csv" >> "runs-ilenia/en/.scores-filtered.txt"
