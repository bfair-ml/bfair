models=( "gpt-oss-20b_databias_generated" "Llama-3.2-1B-Instruct_databias_generated" "Llama-3.2-3B-Instruct_databias_generated" "Mistral-7B-Instruct-v0.3_databias_generated" "Qwen3-4B_databias_generated" "Qwen3-8B_databias_generated" "salamandra-2b-instruct_databias_generated" "salamandra-7b-instruct_databias_generated" )

languages=("spanish" "valencian" "english")

for model in "${models[@]}"; do
  for i in "${!languages[@]}"; do
    for n in all {1..25}; do
      theme_id="${n}.0"
      if [[ "$n" == "all" ]]; then
        theme_id="all"
        subtheme_ids=("all")
      else
        theme_id="${n}.0"
        subtheme_ids=("all" "${n}.1" "${n}.2" "${n}.3" "${n}.4")
      fi
      for subtheme_id in "${subtheme_ids[@]}"; do
        language="${languages[$i]}"

        # IFS='/' read -r a b c model language theme_id subtheme_id <<< ""./runs/runs-rhopa64/salamandra-2b-instruct_databias_generated/english/22.0/22.3"" # Set custom values for testing

        path_prefix="runs/runs-rhopa64/${model}/${language}/${theme_id}/${subtheme_id}"

        echo "============= Model: ${model}, Language: ${language}, Theme ID: ${theme_id}, Subtheme ID: ${subtheme_id} ============="

        mkdir -p "${path_prefix}"
        python -m experiments.bscore --context continuous --dataset "rhopa64|language:${language}|model:${model}|theme_id:${theme_id}|subtheme_id:${subtheme_id}" --export-csv "${path_prefix}/aggregation.csv" --export-scores "${path_prefix}/.scores.json" --use-root yes --lower-proper-nouns yes --semantic-check no --split-endings no   --inspect-professions no --use-morph no >> "${path_prefix}/.log.txt"
        
        if [[ $(cat "${path_prefix}/aggregation.csv") == "words,pos,count_disparity,log_score" ]]; then
          echo "Skipping further processing as the output file contains only the header."
          continue
        fi
        python -m experiments.bscore.zfilter --path "${path_prefix}/aggregation.csv" --zscore-threshold 2
        python -m experiments.bscore.filterby --path "${path_prefix}/aggregation-z2filtered.csv" --pos ADJ
        python -m experiments.bscore.filterby --path "${path_prefix}/aggregation-z2filtered.csv" --pos VERB
        python -m experiments.bscore.disparityfilter --path "${path_prefix}/aggregation.csv"
        python -m experiments.bscore.filterby --path "${path_prefix}/aggregation-disparity-filtered-(0.5).csv" --pos ADJ
        python -m experiments.bscore.filterby --path "${path_prefix}/aggregation-disparity-filtered-(0.5).csv" --pos VERB
        python -m experiments.bscore.filterprofessions --path "${path_prefix}/aggregation.csv" --language "${language}"
        python -m experiments.bscore.getstats --path "${path_prefix}/aggregation-disparity-filtered-(0.5).csv" --output "${path_prefix}/.scores-filtered.json"
        python -m experiments.bscore.getstats --path "${path_prefix}/aggregation.csv" --output "${path_prefix}/.scores-with-confidence.json"

      done
    done
  done
done
