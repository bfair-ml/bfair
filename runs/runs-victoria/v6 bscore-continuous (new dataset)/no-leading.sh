python -m experiments.bscore --context continuous --dataset victoria-GPT3.5-independent --export-csv "runs-victoria/gpt3.5-independent.csv" --use-root yes --lower-proper-nouns yes --semantic-check no --split-endings yes --inspect-professions yes
python -m experiments.bscore --context continuous --dataset victoria-GPT4o-independent --export-csv "runs-victoria/gpt4o-independent.csv" --use-root yes --lower-proper-nouns yes --semantic-check no --split-endings yes --inspect-professions yes
python -m experiments.bscore --context continuous --dataset victoria-Llama_3-independent --export-csv "runs-victoria/llama3-independent.csv" --use-root yes --lower-proper-nouns yes --semantic-check no --split-endings yes --inspect-professions yes
python -m experiments.bscore --context continuous --dataset victoria-Mistral8x7b-independent --export-csv "runs-victoria/mistral8x7b-independent.csv" --use-root yes --lower-proper-nouns yes --semantic-check no --split-endings yes --inspect-professions yes
python -m experiments.bscore --context continuous --dataset victoria-Gemini_1.5-independent --export-csv "runs-victoria/gemini1.5-independent.csv" --use-root yes --lower-proper-nouns yes --semantic-check no --split-endings yes --inspect-professions yes

#

python -m experiments.bscore.zfilter --path runs-victoria/gpt3.5-independent.csv
python -m experiments.bscore.zfilter --path runs-victoria/gpt4o-independent.csv
python -m experiments.bscore.zfilter --path runs-victoria/llama3-independent.csv
python -m experiments.bscore.zfilter --path runs-victoria/mistral8x7b-independent.csv
python -m experiments.bscore.zfilter --path runs-victoria/gemini1.5-independent.csv

python -m experiments.bscore.zfilter --path runs-victoria/gpt3.5-independent.csv --zscore-threshold 2
python -m experiments.bscore.zfilter --path runs-victoria/gpt4o-independent.csv --zscore-threshold 2
python -m experiments.bscore.zfilter --path runs-victoria/llama3-independent.csv --zscore-threshold 2
python -m experiments.bscore.zfilter --path runs-victoria/mistral8x7b-independent.csv --zscore-threshold 2
python -m experiments.bscore.zfilter --path runs-victoria/gemini1.5-independent.csv --zscore-threshold 2

# 

python -m experiments.bscore.filterby --path "runs-victoria/gpt3.5-independent-z3filtered.csv" --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/gpt4o-independent-z3filtered.csv" --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/llama3-independent-z3filtered.csv" --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/mistral8x7b-independent-z3filtered.csv" --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/gemini1.5-independent-z3filtered.csv" --language spanish --pos ADJ  --semantic-model no

python -m experiments.bscore.filterby --path runs-victoria/gpt3.5-independent-z2filtered.csv --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path runs-victoria/gpt4o-independent-z2filtered.csv --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path runs-victoria/llama3-independent-z2filtered.csv --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path runs-victoria/mistral8x7b-independent-z2filtered.csv --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path runs-victoria/gemini1.5-independent-z2filtered.csv --language spanish --pos ADJ  --semantic-model no

# 

python -m experiments.bscore.filterby --path "runs-victoria/gpt3.5-independent-z3filtered.csv" --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/gpt4o-independent-z3filtered.csv" --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/llama3-independent-z3filtered.csv" --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/mistral8x7b-independent-z3filtered.csv" --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/gemini1.5-independent-z3filtered.csv" --language spanish --pos VERB  --semantic-model no

python -m experiments.bscore.filterby --path runs-victoria/gpt3.5-independent-z2filtered.csv --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path runs-victoria/gpt4o-independent-z2filtered.csv --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path runs-victoria/llama3-independent-z2filtered.csv --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path runs-victoria/mistral8x7b-independent-z2filtered.csv --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path runs-victoria/gemini1.5-independent-z2filtered.csv --language spanish --pos VERB  --semantic-model no

# 

python -m experiments.bscore.disparityfilter --path runs-victoria/gpt3.5-independent.csv
python -m experiments.bscore.disparityfilter --path runs-victoria/gpt4o-independent.csv
python -m experiments.bscore.disparityfilter --path runs-victoria/llama3-independent.csv
python -m experiments.bscore.disparityfilter --path runs-victoria/mistral8x7b-independent.csv
python -m experiments.bscore.disparityfilter --path runs-victoria/gemini1.5-independent.csv

# 

python -m experiments.bscore.filterby --path "runs-victoria/gpt3.5-independent-disparity-filtered-(0.5).csv" --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/gpt4o-independent-disparity-filtered-(0.5).csv" --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/llama3-independent-disparity-filtered-(0.5).csv" --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/mistral8x7b-independent-disparity-filtered-(0.5).csv" --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/gemini1.5-independent-disparity-filtered-(0.5).csv" --language spanish --pos ADJ  --semantic-model no

# 

python -m experiments.bscore.filterby --path "runs-victoria/gpt3.5-independent-disparity-filtered-(0.5).csv" --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/gpt4o-independent-disparity-filtered-(0.5).csv" --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/llama3-independent-disparity-filtered-(0.5).csv" --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/mistral8x7b-independent-disparity-filtered-(0.5).csv" --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/gemini1.5-independent-disparity-filtered-(0.5).csv" --language spanish --pos VERB  --semantic-model no

# 

python -m experiments.bscore.filterprofessions --path "runs-victoria/gpt3.5-independent.csv" --language spanish --semantic-model no
python -m experiments.bscore.filterprofessions --path "runs-victoria/gpt4o-independent.csv" --language spanish --semantic-model no
python -m experiments.bscore.filterprofessions --path "runs-victoria/llama3-independent.csv" --language spanish --semantic-model no
python -m experiments.bscore.filterprofessions --path "runs-victoria/mistral8x7b-independent.csv" --language spanish --semantic-model no
python -m experiments.bscore.filterprofessions --path "runs-victoria/gemini1.5-independent.csv" --language spanish --semantic-model no