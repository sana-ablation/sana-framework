#!/usr/bin/env python3
"""Scaffold a benchmark-specific LakeQA transform skill from an approved report."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


REQUIRED_HEADINGS = (
    "## Benchmark artifact inventory",
    "## Recommended benchmark-specific transform skill structure",
)


def _slugify(raw: str) -> str:
    slug = "-".join(re.findall(r"[a-z0-9]+", raw.lower()))
    if not slug:
        raise argparse.ArgumentTypeError("benchmark must contain a letter or digit")
    return slug


def _load_report(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"Report not found: {path}")
    return path.read_text(encoding="utf-8")


def _validate_report(text: str) -> None:
    headings = {line.strip() for line in text.splitlines() if line.startswith("## ")}
    missing = [heading for heading in REQUIRED_HEADINGS if heading not in headings]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Report is missing required heading(s): {joined}")


def _skill_text(slug: str) -> str:
    skill_name = f"{slug}-lakeqa-transform"
    title = skill_name.replace("-", " ").title()
    return f"""---
name: {skill_name}
description: Use when Codex needs to convert {slug} benchmark artifacts into LakeQA-style task JSON after an approved conversion report exists.
---

# {title}

## Scope

Read `references/conversion-report.md` before implementing or running this
transform. Do not re-infer the conversion method; the approved conversion report
defines the benchmark artifacts, evidence/source model, LakeQA mapping, k/d/s
bucket proposal, mirroring convention, validation strategy, and known risks.

If the report is incomplete or appears wrong, stop and revise the report through
the benchmark conversion audit process before changing this transform skill.

The report is a conversion-method artifact, not an answer key. Do not copy task
answers, intermediate answers, executable facts, or source rows from the report
into generated LakeQA tasks. Derive every answer from benchmark source artifacts
and verify it from the recorded source path.

## Workflow

1. Read the approved conversion report in `references/conversion-report.md`.
2. Inspect the benchmark examples and source artifacts named by the report.
3. Implement or run only the benchmark-to-LakeQA transform described there.
4. Generate LakeQA task JSON and mirrored source artifacts using the report's
   naming, bucketing, and validation guidance.
5. Validate the generated LakeQA tasks against the report's validation strategy.
6. Maintain a structured skipped/error log for unpromotable examples.
7. Defer ideal profile, ideal code, and verifier handoff until LakeQA tasks exist.

## Batch Flow

Process a small batch first. A good default is five examples, or fewer if the
report recommends a smaller audit batch.

For each batch:

1. Select examples according to the report's sampling and promotion rules.
2. Inspect LakeQA exemplars for JSON shape and reasoning-hop style only.
3. Convert only examples that are fair, source-grounded, and verifiable.
4. Write skipped examples to a structured error log with precise reasons.
5. Run the validation checklist before claiming the batch is ready.

## Worker Assignment Template

Use this prompt when delegating one example at a time:

```text
You are converting exactly one benchmark example into LakeQA-style task JSON.
You are not alone in the codebase; do not revert or overwrite edits made by
others.

Approved conversion report:
<absolute path to references/conversion-report.md>

Input example/source artifacts:
<absolute paths from the approved report>

Reference LakeQA examples:
<absolute path to LakeQA exemplar tasks>

Write only:
<absolute path to temp converted task json>
or
<absolute path to temp error json>

Requirements:
- Follow the approved report; do not re-infer the conversion method.
- Use LakeQA examples as shape/style exemplars only.
- Do not copy answers, intermediate answers, executable facts, or source rows
  from the report or exemplars.
- Derive every node answer from benchmark source artifacts.
- Keep subquestions source-agnostic and necessary for the final answer.
- Verify executable facts against node answers before writing success.
- If the example is unfair, ungrounded, or unverifiable, write a structured
  skip reason instead of forcing a task.
```

## Final Task Shape

Use the final JSON shape required by the approved report and local LakeQA
exemplars. Unless the report specifies a stricter schema, each converted task
should include:

```text
question
answer
nodes
reasoning_chain
datasets_used
reasoning_hops
_provenance
```

Every final question should preserve the local LakeQA answer-format convention,
usually ending with:

```text
Write your answer as [ANSWER].
```

## k/d/s Rules

Use the approved report's topology rule. If it follows the standard LakeQA
runtime-profile convention:

```text
k = number of reasoning hops
d = maximum fanout of any reasoning hop
s = number of unique final task sources/datasets used
```

Never copy import-stage or benchmark-native axes into final bucket names unless
the approved report explicitly defines them as LakeQA runtime axes.

## Provenance

Carry enough provenance for every converted task to be audited later:

```text
benchmark name
source example id
source split/domain/difficulty when available
local authoring source path for each node
mirrored source ref for each final node
promotion/conversion status
conversion rule or report version
```

Provenance may identify source artifacts, but it must not store hidden answers,
solution snippets, or answer-bearing rows as shortcuts.

## Fairness And Leakage Requirements

Promote an example only when:

- The final answer is derivable from mirrored source artifacts.
- Each intermediate answer is necessary and feeds later reasoning.
- Node facts compute answers from data, not from hard-coded answer strings.
- Source refs are discoverable through the intended tool surface.
- The final task does not reveal the answer through provenance, filenames,
  reasoning text, source ids, or ideal-artifact fields.

Skip an example when these requirements cannot be met.

## Validation Checklist

For each generated task:

```text
bucket name matches approved k/d/s rule
question uses the required answer-format suffix
every reasoning_hops node id exists
every node source maps to a mirrored source artifact
datasets_used matches node sources
facts do not hard-code final or intermediate answers
facts execute and reproduce node.answer when executable facts are required
terminal answer equals task.answer
_provenance contains source mapping but no answer-bearing shortcuts
no ideal profile, ideal query, or ideal code is authored before LakeQA tasks exist
```

## Handoff Boundary

After converted LakeQA tasks exist, use `author-ideal-profiles`,
`profile-verifier`, and `author-ideal-code` as appropriate. Do not duplicate
those workflows inside this transform skill.
"""


def scaffold(report: Path, benchmark: str, output_root: Path, *, force: bool = False) -> Path:
    text = _load_report(report)
    _validate_report(text)

    slug = _slugify(benchmark)
    skill_root = output_root / f"{slug}-lakeqa-transform"
    if skill_root.exists() and not force:
        raise SystemExit(f"Skill directory already exists: {skill_root}")
    references = skill_root / "references"
    references.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(report, references / "conversion-report.md")
    (skill_root / "SKILL.md").write_text(_skill_text(slug), encoding="utf-8")
    return skill_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--benchmark", required=True, type=_slugify)
    parser.add_argument("--output-root", default="sana-profiling/skills")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing generated skill directory",
    )
    args = parser.parse_args()

    report = Path(args.report).expanduser().resolve()
    output_root = Path(args.output_root).expanduser()
    scaffold(report, args.benchmark, output_root, force=args.force)


if __name__ == "__main__":
    main()
