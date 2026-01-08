#!/bin/bash

# Set the directory to list files from
directory="datasets/ilenia/en"

# Loop through each file in the directory
for file in "$directory"/*; do
    # Check if it is a regular file (not a directory)
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        mkdir "runs-ilenia/en/$filename"

        # Call the other script, passing the file name as a parameter        
        python -m experiments.bscore --context continuous --dataset "ilenia|path:$file|annotated:no" --export-csv "runs-ilenia/en/$filename/individual.csv" --use-root yes --lower-proper-nouns yes --semantic-check no --split-endings no --inspect-professions no --use-morph no >> "runs-ilenia/en/$filename/.log.txt"
        #
        python -m experiments.bscore.zfilter --path "runs-ilenia/en/$filename/individual.csv" --zscore-threshold 2
        # 
        python -m experiments.bscore.filterby --path "runs-ilenia/en/$filename/individual-z2filtered.csv" --pos ADJ
        # 
        python -m experiments.bscore.filterby --path "runs-ilenia/en/$filename/individual-z2filtered.csv" --pos VERB
        # 
        python -m experiments.bscore.disparityfilter --path "runs-ilenia/en/$filename/individual.csv"
        # 
        python -m experiments.bscore.filterby --path "runs-ilenia/en/$filename/individual-disparity-filtered-(0.5).csv" --pos ADJ
        # 
        python -m experiments.bscore.filterby --path "runs-ilenia/en/$filename/individual-disparity-filtered-(0.5).csv" --pos VERB
        # 
        python -m experiments.bscore.getstats --path "runs-ilenia/en/$filename/individual-disparity-filtered-(0.5).csv" >> "runs-ilenia/en/$filename/.scores-filtered.txt"
    fi
done