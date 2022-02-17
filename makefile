dashboard:
	streamlit run --browser.serverAddress localhost dashboard.py

dashboard-nowatch:
	streamlit run --browser.serverAddress localhost --server.fileWatcherType none dashboard.py