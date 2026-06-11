# HotpotQA Generated Conversion Run

This folder is a repo-local five-import smoke run of the two-skill benchmark conversion workflow.

It demonstrates:

- benchmark artifact sampling with bucket/provenance diversity
- redacted conversion report authoring
- generated benchmark-specific transform skill scaffolding
- five HotpotQA mini tasks converted into the current `benchmarks/<benchmark>/tasks-mini/{tasks,runtime-profiles}` shape
- ideal runtime profile authoring for each converted task
- text-evidence ideal-code checks that keep `ideal_code` and `ideal_query` empty
- prompt/report/runtime-profile leakage checks

See `run-log.md` and `validation.json` for the execution summary.
