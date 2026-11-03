# Portfolio Diagnostic (modular)

Decision support for one investor's Indian listed-equity portfolio. Not investment advice; it never issues orders.

    app.py                    Streamlit page: upload your files, answer the forms, download report + workbook + carry-over pack
    portfolio_diagnostic.py   command line / IDLE entry point (reads files from disk, writes HTML + Excel)
    pdiag/                    the engine, split into small modules (see pdiag/__init__.py); appcore.py holds the page's helpers
    scoring.py, *_engine.py   the factor engine files (stay next to portfolio_diagnostic.py and app.py)
    tests/, check_step27.py   synthetic-data tests: old single file and new modules behave the same
    check_step28.py           synthetic-data tests for the page and its helpers
    step26_spike_app.py       hosting test page (does Yahoo / NSE answer from the host?)
    check_repo_safe.py        run before every commit: refuses if private files are in the folder
    broker_inspect.py         prints the masked shape of a broker statement
    legacy/                   the single-file version the modules were cut from (reference for check_step27.py)

The Tkinter pop-ups of the local version (input_assistant.py) are not in this repository; the page has its own forms.
Never commit holdings, broker statements, evidence/cost/buy-date files, reports or carry-over files.
The app keeps nothing of yours: you upload at the start of a run and download the results and the carry-over pack at the end.
Deploy on Streamlit: main file path app.py; requirements.txt pins the versions the hosting test ran on.
