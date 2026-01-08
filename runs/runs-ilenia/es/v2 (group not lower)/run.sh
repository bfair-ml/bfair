python -m experiments.bscore --context continuous --dataset ilenia --export-csv "runs-ilenia/aggregation.csv" --use-root yes --lower-proper-nouns no --semantic-check no --split-endings no --inspect-professions no --use-morph no >> "runs-ilenia/.log.txt"

#

python -m experiments.bscore.zfilter --path runs-ilenia/aggregation.csv --zscore-threshold 2

# 

python -m experiments.bscore.filterby --path "runs-ilenia/aggregation-z2filtered.csv" --pos ADJ

# 

python -m experiments.bscore.filterby --path "runs-ilenia/aggregation-z2filtered.csv" --pos VERB

# 

python -m experiments.bscore.disparityfilter --path runs-ilenia/aggregation.csv

# 

python -m experiments.bscore.filterby --path "runs-ilenia/aggregation-disparity-filtered-(0.5).csv" --pos ADJ

# 

python -m experiments.bscore.filterby --path "runs-ilenia/aggregation-disparity-filtered-(0.5).csv" --pos VERB

# 

python -m experiments.bscore.filterprofessions --path "runs-ilenia/aggregation.csv" --language spanish

# 

python -m experiments.bscore.getstats --path "runs-ilenia/aggregation-disparity-filtered-(0.5).csv" >> "runs-ilenia/.scores-filtered.txt"
