"""pdiag - the Portfolio Diagnostic engine, split into small modules (Step 27, rebuilt 3 Nov 2026).

  settings, helpers   config, errors, utils
  data in             providers, evidence, fundamentals, loaders
  scoring, decisions  engine_link, factors, actions, register
  seams               hooks (pop-ups / forms), state (carry-over files or memory)
  app mode            appcore (page helpers, carry-over pack), resilient (paced company-info calls, last good copy, reporting)
  run                 pipeline: stage_evidence -> hooks.prompts -> stage_prices -> apply_cost_review -> stage_analysis
  output              report_html, report_excel
  proof               verify (used by check_step27.py)

Library code never calls sys.exit and never opens a pop-up or a browser; entry points do.
Decision support only; not investment advice."""
