log = ./log.txt
lookback = 5

dashboard:
	streamlit run --browser.serverAddress localhost dashboard.py

dashboard-nowatch:
	streamlit run --browser.serverAddress localhost --server.fileWatcherType none dashboard.py

show-improvement:
	cat ${log} | grep "Best solution"

show-improvement-params:
	cat ${log} | grep -B ${lookback} "Best solution"

show-generations:
	cat ${log} | grep "best_fn"

watch:
	tail -f ${log}