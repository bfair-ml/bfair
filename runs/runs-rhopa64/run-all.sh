models=( "gpt-oss-20b_databias_generated" "Llama-3.2-1B-Instruct_databias_generated" "Llama-3.2-3B-Instruct_databias_generated" "Mistral-7B-Instruct-v0.3_databias_generated" "Qwen3-4B_databias_generated" "Qwen3-8B_databias_generated" "salamandra-2b-instruct_databias_generated" "salamandra-7b-instruct_databias_generated" )

languages=("spanish" "valencian" "english")

for model in "${models[@]}"; do
  for i in "${!languages[@]}"; do
    language="${languages[$i]}"

    mkdir -p "runs/runs-rhopa64/${model}/${language}"
    python -m experiments.bscore --context continuous --dataset "rhopa64|language:${language}|model:${model}" --export-csv "runs/runs-rhopa64/${model}/${language}/aggregation.csv" --export-scores "runs/runs-rhopa64/${model}/${language}/.scores.json" --use-root yes --lower-proper-nouns yes --semantic-check no --split-endings no   --inspect-professions no --use-morph no >> "runs/runs-rhopa64/${model}/${language}/.log.txt"
    python -m experiments.bscore.zfilter --path "runs/runs-rhopa64/${model}/${language}/aggregation.csv" --zscore-threshold 2
    python -m experiments.bscore.filterby --path "runs/runs-rhopa64/${model}/${language}/aggregation-z2filtered.csv" --pos ADJ
    python -m experiments.bscore.filterby --path "runs/runs-rhopa64/${model}/${language}/aggregation-z2filtered.csv" --pos VERB
    python -m experiments.bscore.disparityfilter --path "runs/runs-rhopa64/${model}/${language}/aggregation.csv"
    python -m experiments.bscore.filterby --path "runs/runs-rhopa64/${model}/${language}/aggregation-disparity-filtered-(0.5).csv" --pos ADJ
    python -m experiments.bscore.filterby --path "runs/runs-rhopa64/${model}/${language}/aggregation-disparity-filtered-(0.5).csv" --pos VERB
    python -m experiments.bscore.filterprofessions --path "runs/runs-rhopa64/${model}/${language}/aggregation.csv" --language "${language}"
    python -m experiments.bscore.getstats --path "runs/runs-rhopa64/${model}/${language}/aggregation-disparity-filtered-(0.5).csv" > "runs/runs-rhopa64/${model}/${language}/.scores-filtered.txt"
  done
done
