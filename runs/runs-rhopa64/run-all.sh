theme_ids=("all" "1.0" "2.0" "3.0" "4.0" "5.0" "6.0" "7.0" "8.0" "9.0" "10.0" "11.0" "12.0" "13.0" "14.0" "15.0" "16.0" "17.0" "18.0" "19.0" "20.0" "21.0" "22.0" "23.0" "24.0" "25.0")

subtheme_ids=("all" "1.1" "1.2" "1.3" "1.4" "2.1" "2.2" "2.3" "2.4" "3.1" "3.2" "3.3" "3.4" "4.1" "4.2" "4.3" "4.4" "5.1" "5.2" "5.3" "5.4" "6.1" "6.2" "6.3" "6.4" "7.1" "7.2" "7.3" "7.4" "8.1" "8.2" "8.3" "8.4" "9.1" "9.2" "9.3" "9.4" "10.1" "10.2" "10.3" "10.4" "11.1" "11.2" "11.3" "11.4" "12.1" "12.2" "12.3" "12.4" "13.1" "13.2" "13.3" "13.4" "14.1" "14.2" "14.3" "14.4" "15.1" "15.2" "15.3" "15.4" "16.1" "16.2" "16.3" "16.4" "17.1" "17.2" "17.3" "17.4" "18.1" "18.2" "18.3" "18.4" "19.1" "19.2" "19.3" "19.4" "20.1" "20.2" "20.3" "20.4" "21.1" "21.2" "21.3" "21.4" "22.1" "22.2" "22.3" "22.4" "23.1" "23.2" "23.3" "23.4" "24.1" "24.2" "24.3" "24.4" "25.1" "25.2" "25.3" "25.4")

models=( "gpt-oss-20b_databias_generated" "Llama-3.2-1B-Instruct_databias_generated" "Llama-3.2-3B-Instruct_databias_generated" "Mistral-7B-Instruct-v0.3_databias_generated" "Qwen3-4B_databias_generated" "Qwen3-8B_databias_generated" "salamandra-2b-instruct_databias_generated" "salamandra-7b-instruct_databias_generated" )

languages=("spanish" "valencian" "english")

for model in "${models[@]}"; do
  for i in "${!languages[@]}"; do
    for theme_id in "${theme_ids[@]}"; do
      for subtheme_id in "${subtheme_ids[@]}"; do

        language="${languages[$i]}"
        path_prefix="runs/runs-rhopa64/${model}/${language}/${theme_id}/${subtheme_id}"

        mkdir -p "${path_prefix}"
        python -m experiments.bscore --context continuous --dataset "rhopa64|language:${language}|model:${model}|theme_id:${theme_id}|subtheme_id:${subtheme_id}" --export-csv "${path_prefix}/aggregation.csv" --export-scores "${path_prefix}/.scores.json" --use-root yes --lower-proper-nouns yes --semantic-check no --split-endings no   --inspect-professions no --use-morph no >> "${path_prefix}/.log.txt"
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
