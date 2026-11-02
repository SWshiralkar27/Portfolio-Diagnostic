# Portfolio Diagnostic (modular)

Decision support for one investor's Indian listed-equity portfolio. Not investment advice; it never issues orders.

    portfolio_diagnostic.py   command line / IDLE entry point (reads files from disk, writes HTML + Excel)
    pdiag/                    the engine, split into small modules (see pdiag/__init__.py for the layers)
    scoring.py, *_engine.py   the factor engine files (stay next to portfolio_diagnostic.py)
    input_assistant.py        Tkinter pop-ups for the local version
    public_data/              company-level public data only (added in a later step)
    tests/, check_step27.py   synthetic-data tests; run check_step27.py to prove old and new behave the same
    check_repo_safe.py        run before every commit: refuses if private files are in the folder
    broker_inspect.py         prints the masked shape of a broker statement

Never commit holdings, broker statements, evidence/cost/buy-date files, reports or carry-over files.
The app keeps nothing of yours: you upload at the start of a run and download the results and carry-over files at the end.
