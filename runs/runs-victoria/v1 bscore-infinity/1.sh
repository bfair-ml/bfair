python -m experiments.bscore --dataset victoria-GPT3.5 --export-csv "runs-victoria/gpt3.5.csv" --use-root yes

# ## count_disparity
# - **Mean** 0.7258171339917716
# - **Standard Deviation** 0.25810415672419734
# ## log_score
# - **Mean** -0.09319702045114005
# - **Standard Deviation** 1.1193284728545043


python -m experiments.bscore --dataset victoria-GPT4o --export-csv "runs-victoria/gpt4o.csv" --use-root yes

# ## count_disparity (male then female)
# - **Mean** 0.6864777573764791
# - **Standard Deviation** 0.29688513777548503
# ## log_score (male then female)
# - **Mean** -0.07651276324809307
# - **Standard Deviation** 1.229464790825561


python -m experiments.bscore --dataset victoria-Llama_3 --export-csv "runs-victoria/llama3.csv" --use-root yes

# ## count_disparity (male then female)
# - **Mean** 0.6864777573764791
# - **Standard Deviation** 0.29688513777548503
# ## log_score (male then female)
# - **Mean** -0.07651276324809307
# - **Standard Deviation** 1.229464790825561

python -m experiments.bscore --dataset victoria-Mistral8x7b --export-csv "runs-victoria/mistral8x7b.csv" --use-root yes

# ## count_disparity (male then female)
# - **Mean** 0.6778418019873536
# - **Standard Deviation** 0.27955188894351213
# ## log_score (male then female)
# - **Mean** 0.09978916741907921
# - **Standard Deviation** 1.2259160876281645


python -m experiments.bscore --dataset victoria-Gemini_1.5 --export-csv "runs-victoria/gemini1.5.csv" --use-root yes

# ## count_disparity (male then female)
# - **Mean** 0.6729434504532555
# - **Standard Deviation** 0.30995714025491755
# ## log_score (male then female)
# - **Mean** -0.059774356566717396
# - **Standard Deviation** 1.0398362663947072