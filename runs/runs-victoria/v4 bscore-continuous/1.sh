python -m experiments.bscore --context continuous --dataset victoria-GPT3.5 --export-csv "runs-victoria/gpt3.5.csv" --use-root yes --lower-proper-nouns yes
python -m experiments.bscore --context continuous --dataset victoria-GPT4o --export-csv "runs-victoria/gpt4o.csv" --use-root yes --lower-proper-nouns yes
python -m experiments.bscore --context continuous --dataset victoria-Llama_3 --export-csv "runs-victoria/llama3.csv" --use-root yes --lower-proper-nouns yes
python -m experiments.bscore --context continuous --dataset victoria-Mistral8x7b --export-csv "runs-victoria/mistral8x7b.csv" --use-root yes --lower-proper-nouns yes
python -m experiments.bscore --context continuous --dataset victoria-Gemini_1.5 --export-csv "runs-victoria/gemini1.5.csv" --use-root yes --lower-proper-nouns yes