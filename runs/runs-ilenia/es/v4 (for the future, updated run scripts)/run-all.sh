python -m experiments.bscore --context continuous --dataset "ilenia|language:spanish" --export-csv "runs-ilenia/es/aggregation.csv" --use-root yes --lower-proper-nouns yes --semantic-check no --split-endings no --inspect-professions no --use-morph no >> "runs-ilenia/es/.log.txt"

#

python -m experiments.bscore.zfilter --path runs-ilenia/es/aggregation.csv --zscore-threshold 2

# 

python -m experiments.bscore.filterby --path "runs-ilenia/es/aggregation-z2filtered.csv" --pos ADJ

# 

python -m experiments.bscore.filterby --path "runs-ilenia/es/aggregation-z2filtered.csv" --pos VERB

# 

python -m experiments.bscore.disparityfilter --path runs-ilenia/es/aggregation.csv

# 

python -m experiments.bscore.filterby --path "runs-ilenia/es/aggregation-disparity-filtered-(0.5).csv" --pos ADJ

# 

python -m experiments.bscore.filterby --path "runs-ilenia/es/aggregation-disparity-filtered-(0.5).csv" --pos VERB

# 

python -m experiments.bscore.filterprofessions --path "runs-ilenia/es/aggregation.csv" --language spanish

# 

python -m experiments.bscore.getstats --path "runs-ilenia/es/aggregation-disparity-filtered-(0.5).csv" >> "runs-ilenia/es/.scores-filtered.txt"
