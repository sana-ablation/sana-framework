"""Classify *why* wrong answers were wrong, using external model calls.

Two passes, in order:
  1. audit_runner  - audit one eval CSV; a second model validates each row
  2. rerun_queue   - scan the tree for rows that failed validation, re-audit them

Then: report -> combine_grouped_models -> the two paper PDFs.

migrate_taxonomy is a one-shot migration of older audit outputs to the
current taxonomy, not part of the routine pass.
"""
