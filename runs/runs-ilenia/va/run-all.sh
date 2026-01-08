python -m experiments.bscore --context continuous --dataset "ilenia|language:valencian" --export-csv "runs/runs-ilenia/va/aggregation.csv" --use-root yes --lower-proper-nouns yes --semantic-check no --split-endings no --inspect-professions no --use-morph no >> "runs/runs-ilenia/va/.log.txt"

#

python -m experiments.bscore.zfilter --path runs/runs-ilenia/va/aggregation.csv --zscore-threshold 2

# 

python -m experiments.bscore.filterby --path "runs/runs-ilenia/va/aggregation-z2filtered.csv" --pos ADJ

# 

python -m experiments.bscore.filterby --path "runs/runs-ilenia/va/aggregation-z2filtered.csv" --pos VERB

# 

python -m experiments.bscore.disparityfilter --path runs/runs-ilenia/va/aggregation.csv

# 

python -m experiments.bscore.filterby --path "runs/runs-ilenia/va/aggregation-disparity-filtered-(0.5).csv" --pos ADJ

# 

python -m experiments.bscore.filterby --path "runs/runs-ilenia/va/aggregation-disparity-filtered-(0.5).csv" --pos VERB

# 

python -m experiments.bscore.filterprofessions --path "runs/runs-ilenia/va/aggregation.csv" --language valencian

# 

python -m experiments.bscore.getstats --path "runs/runs-ilenia/va/aggregation-disparity-filtered-(0.5).csv" >> "runs/runs-ilenia/va/.scores-filtered.txt"
