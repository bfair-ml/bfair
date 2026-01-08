python -m experiments adult adult-runs-sp 1 \
--iterations 1000 \
--n-classifiers 20 \
--popsize 50 \
--diversity "double-fault" \
--token 5044881135:AAEzEPL46_uW88R4V6vUCN9c1obOe2oyuKs \
--channel @bfair_report \
--fairness "statistical-parity" \
--fairness-under "0.1"
