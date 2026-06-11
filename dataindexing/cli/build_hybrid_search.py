"""CLI wrapper for hybrid LanceDB index construction."""

from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_module("dataindexing.hybrid_search.builder", run_name="__main__")


if __name__ == "__main__":
    main()
