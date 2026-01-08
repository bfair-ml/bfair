python -m experiments.bscore --context continuous --dataset victoria-GPT3.5-leading --export-csv "runs-victoria/gpt3.5-leading.csv" --use-root yes --lower-proper-nouns yes --semantic-check no --split-endings yes --inspect-professions yes
python -m experiments.bscore --context continuous --dataset victoria-GPT4o-leading --export-csv "runs-victoria/gpt4o-leading.csv" --use-root yes --lower-proper-nouns yes --semantic-check no --split-endings yes --inspect-professions yes
python -m experiments.bscore --context continuous --dataset victoria-Llama_3-leading --export-csv "runs-victoria/llama3-leading.csv" --use-root yes --lower-proper-nouns yes --semantic-check no --split-endings yes --inspect-professions yes
python -m experiments.bscore --context continuous --dataset victoria-Mistral8x7b-leading --export-csv "runs-victoria/mistral8x7b-leading.csv" --use-root yes --lower-proper-nouns yes --semantic-check no --split-endings yes --inspect-professions yes
python -m experiments.bscore --context continuous --dataset victoria-Gemini_1.5-leading --export-csv "runs-victoria/gemini1.5-leading.csv" --use-root yes --lower-proper-nouns yes --semantic-check no --split-endings yes --inspect-professions yes

#

python -m experiments.bscore.zfilter --path runs-victoria/gpt3.5-leading.csv
python -m experiments.bscore.zfilter --path runs-victoria/gpt4o-leading.csv
python -m experiments.bscore.zfilter --path runs-victoria/llama3-leading.csv
python -m experiments.bscore.zfilter --path runs-victoria/mistral8x7b-leading.csv
python -m experiments.bscore.zfilter --path runs-victoria/gemini1.5-leading.csv

python -m experiments.bscore.zfilter --path runs-victoria/gpt3.5-leading.csv --zscore-threshold 2
python -m experiments.bscore.zfilter --path runs-victoria/gpt4o-leading.csv --zscore-threshold 2
python -m experiments.bscore.zfilter --path runs-victoria/llama3-leading.csv --zscore-threshold 2
python -m experiments.bscore.zfilter --path runs-victoria/mistral8x7b-leading.csv --zscore-threshold 2
python -m experiments.bscore.zfilter --path runs-victoria/gemini1.5-leading.csv --zscore-threshold 2

# 

python -m experiments.bscore.filterby --path "runs-victoria/gpt3.5-leading-z3filtered.csv" --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/gpt4o-leading-z3filtered.csv" --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/llama3-leading-z3filtered.csv" --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/mistral8x7b-leading-z3filtered.csv" --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/gemini1.5-leading-z3filtered.csv" --language spanish --pos ADJ  --semantic-model no

python -m experiments.bscore.filterby --path runs-victoria/gpt3.5-leading-z2filtered.csv --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path runs-victoria/gpt4o-leading-z2filtered.csv --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path runs-victoria/llama3-leading-z2filtered.csv --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path runs-victoria/mistral8x7b-leading-z2filtered.csv --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path runs-victoria/gemini1.5-leading-z2filtered.csv --language spanish --pos ADJ  --semantic-model no

# 

python -m experiments.bscore.filterby --path "runs-victoria/gpt3.5-leading-z3filtered.csv" --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/gpt4o-leading-z3filtered.csv" --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/llama3-leading-z3filtered.csv" --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/mistral8x7b-leading-z3filtered.csv" --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/gemini1.5-leading-z3filtered.csv" --language spanish --pos VERB  --semantic-model no

python -m experiments.bscore.filterby --path runs-victoria/gpt3.5-leading-z2filtered.csv --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path runs-victoria/gpt4o-leading-z2filtered.csv --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path runs-victoria/llama3-leading-z2filtered.csv --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path runs-victoria/mistral8x7b-leading-z2filtered.csv --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path runs-victoria/gemini1.5-leading-z2filtered.csv --language spanish --pos VERB  --semantic-model no

# 

python -m experiments.bscore.disparityfilter --path runs-victoria/gpt3.5-leading.csv
python -m experiments.bscore.disparityfilter --path runs-victoria/gpt4o-leading.csv
python -m experiments.bscore.disparityfilter --path runs-victoria/llama3-leading.csv
python -m experiments.bscore.disparityfilter --path runs-victoria/mistral8x7b-leading.csv
python -m experiments.bscore.disparityfilter --path runs-victoria/gemini1.5-leading.csv

# 

python -m experiments.bscore.filterby --path "runs-victoria/gpt3.5-leading-disparity-filtered-(0.5).csv" --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/gpt4o-leading-disparity-filtered-(0.5).csv" --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/llama3-leading-disparity-filtered-(0.5).csv" --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/mistral8x7b-leading-disparity-filtered-(0.5).csv" --language spanish --pos ADJ  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/gemini1.5-leading-disparity-filtered-(0.5).csv" --language spanish --pos ADJ  --semantic-model no

# 

python -m experiments.bscore.filterby --path "runs-victoria/gpt3.5-leading-disparity-filtered-(0.5).csv" --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/gpt4o-leading-disparity-filtered-(0.5).csv" --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/llama3-leading-disparity-filtered-(0.5).csv" --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/mistral8x7b-leading-disparity-filtered-(0.5).csv" --language spanish --pos VERB  --semantic-model no
python -m experiments.bscore.filterby --path "runs-victoria/gemini1.5-leading-disparity-filtered-(0.5).csv" --language spanish --pos VERB  --semantic-model no

# 

python -m experiments.bscore.filterprofessions --path "runs-victoria/gpt3.5-leading.csv" --language spanish --semantic-model no
python -m experiments.bscore.filterprofessions --path "runs-victoria/gpt4o-leading.csv" --language spanish --semantic-model no
python -m experiments.bscore.filterprofessions --path "runs-victoria/llama3-leading.csv" --language spanish --semantic-model no
python -m experiments.bscore.filterprofessions --path "runs-victoria/mistral8x7b-leading.csv" --language spanish --semantic-model no
python -m experiments.bscore.filterprofessions --path "runs-victoria/gemini1.5-leading.csv" --language spanish --semantic-model no