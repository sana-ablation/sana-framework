import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SANA_PROFILING = ROOT / "sana-profiling"


def test_sana_profiling_folder_has_public_index():
    readme = SANA_PROFILING / "README.md"

    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")
    assert "Benchmark conversion skills" in text
    assert "benchmark-lakeqa-conversion-auditor" in text
    assert "benchmark-lakeqa-skill-scaffolder" in text


AUDITOR = SANA_PROFILING / "skills" / "benchmark-lakeqa-conversion-auditor"
SCAFFOLDER = SANA_PROFILING / "skills" / "benchmark-lakeqa-skill-scaffolder"
RUNS = SANA_PROFILING / "runs"


def test_conversion_auditor_skill_contract():
    skill = AUDITOR / "SKILL.md"
    template = AUDITOR / "references" / "report-template.md"

    assert skill.is_file()
    text = skill.read_text(encoding="utf-8")
    assert "name: benchmark-lakeqa-conversion-auditor" in text
    assert "docs/benchmark-conversions/<benchmark>-lakeqa-conversion.md" in text
    assert "structural diversity first" in text
    assert "author-ideal-profiles" in text
    assert "profile-verifier" in text
    assert "author-ideal-code" in text
    assert "Do not paste" in text
    assert "answer key" in text

    assert template.is_file()
    template_text = template.read_text(encoding="utf-8")
    for heading in [
        "Benchmark artifact inventory",
        "Auto-sampled examples and rationale",
        "Evidence/source model",
        "LakeQA task mapping",
        "Ideal artifact feasibility",
        "Recommended benchmark-specific transform skill structure",
    ]:
        assert f"## {heading}" in template_text


def test_sample_benchmark_artifacts_prefers_structural_diversity(tmp_path):
    root = tmp_path / "benchmark"
    root.mkdir()
    (root / "one.json").write_text(
        json.dumps({"question": "q1", "answer": "a1", "evidence": ["doc1"]}),
        encoding="utf-8",
    )
    (root / "two.json").write_text(
        json.dumps({"question": "q2", "answer": "a2", "evidence": ["doc1", "doc2"]}),
        encoding="utf-8",
    )
    (root / "notes.txt").write_text("not json", encoding="utf-8")

    script = AUDITOR / "scripts" / "sample_benchmark_artifacts.py"
    result = subprocess.run(
        [sys.executable, str(script), str(root), "--limit", "2"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["benchmark_root"] == str(root)
    assert len(payload["samples"]) == 2
    signatures = {sample["signature"] for sample in payload["samples"]}
    assert any("evidence:list:1" in signature for signature in signatures)
    assert any("evidence:list:2" in signature for signature in signatures)


def test_sample_benchmark_artifacts_includes_common_tabular_and_jsonl_formats(tmp_path):
    root = tmp_path / "benchmark"
    root.mkdir()
    (root / "tasks.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"question": "q1", "answer": "a1"}),
                json.dumps({"question": "q2", "answer": "a2", "evidence": ["doc"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "metadata.csv").write_text(
        "question,answer,split\nq3,a3,dev\nq4,a4,test\n",
        encoding="utf-8",
    )
    (root / "events.ndjson").write_text(
        json.dumps({"question": "q5", "answer": "a5"}) + "\n",
        encoding="utf-8",
    )
    (root / "table.parquet").write_bytes(b"PAR1")

    script = AUDITOR / "scripts" / "sample_benchmark_artifacts.py"
    result = subprocess.run(
        [sys.executable, str(script), str(root), "--limit", "6"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    relative_paths = {sample["relative_path"] for sample in payload["samples"]}
    assert "tasks.jsonl" in relative_paths
    assert "metadata.csv" in relative_paths
    assert "events.ndjson" in relative_paths
    assert "table.parquet" in relative_paths
    assert payload["candidate_count"] == 4


def test_sample_benchmark_artifacts_prefers_bucket_and_provenance_diversity(tmp_path):
    root = tmp_path / "benchmark"
    first_bucket = root / "k-2-d-1"
    second_bucket = root / "k-2-d-2"
    first_bucket.mkdir(parents=True)
    second_bucket.mkdir(parents=True)
    base = {
        "question": "q",
        "answer": "a",
        "nodes": {"n1": {"answer": "a"}},
        "reasoning_hops": [{"node_ids": ["n1"]}],
    }
    first = dict(base, _provenance={"benchmark": "hotpotqa", "type": "comparison"})
    second = dict(base, _provenance={"benchmark": "hotpotqa", "type": "bridge"})
    (first_bucket / "task_1.json").write_text(json.dumps(first), encoding="utf-8")
    (second_bucket / "task_1.json").write_text(json.dumps(second), encoding="utf-8")

    script = AUDITOR / "scripts" / "sample_benchmark_artifacts.py"
    result = subprocess.run(
        [sys.executable, str(script), str(root), "--limit", "2"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert {sample["bucket"] for sample in payload["samples"]} == {
        "k-2-d-1",
        "k-2-d-2",
    }
    assert {sample["metadata_key"] for sample in payload["samples"]} == {
        "benchmark:hotpotqa|type:comparison",
        "benchmark:hotpotqa|type:bridge",
    }


def test_sample_benchmark_artifacts_rejects_non_positive_limit(tmp_path):
    root = tmp_path / "benchmark"
    root.mkdir()
    (root / "one.json").write_text(json.dumps({"question": "q1"}), encoding="utf-8")

    script = AUDITOR / "scripts" / "sample_benchmark_artifacts.py"
    result = subprocess.run(
        [sys.executable, str(script), str(root), "--limit", "0"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "must be a positive integer" in result.stderr


def test_scaffolder_skill_contract():
    skill = SCAFFOLDER / "SKILL.md"

    assert skill.is_file()
    text = skill.read_text(encoding="utf-8")
    assert "name: benchmark-lakeqa-skill-scaffolder" in text
    assert "approved conversion report" in text
    assert "must not invent" in text


def test_scaffold_benchmark_skill_creates_transform_skill(tmp_path):
    report = tmp_path / "demo-report.md"
    report.write_text(
        "\n".join(
            [
                "# Demo LakeQA Conversion Report",
                "",
                "## Benchmark artifact inventory",
                "",
                "Demo inventory.",
                "",
                "## Recommended benchmark-specific transform skill structure",
                "",
                "Demo transform structure.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    script = SCAFFOLDER / "scripts" / "scaffold_benchmark_skill.py"
    output_root = tmp_path / "skills"
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(report),
            "--benchmark",
            "Demo Benchmark!",
            "--output-root",
            str(output_root),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    generated_root = output_root / "demo-benchmark-lakeqa-transform"
    skill = generated_root / "SKILL.md"
    copied_report = generated_root / "references" / "conversion-report.md"

    assert skill.is_file()
    assert copied_report.is_file()
    assert copied_report.read_text(encoding="utf-8") == report.read_text(encoding="utf-8")
    skill_text = skill.read_text(encoding="utf-8")
    assert "name: demo-benchmark-lakeqa-transform" in skill_text
    assert "Do not re-infer the conversion method" in skill_text
    for section in [
        "## Batch Flow",
        "## Worker Assignment Template",
        "## Final Task Shape",
        "## k/d/s Rules",
        "## Provenance",
        "## Fairness And Leakage Requirements",
        "## Validation Checklist",
    ]:
        assert section in skill_text
    assert "Do not copy answers" in skill_text
    assert "no answer-bearing shortcuts" in skill_text


def test_scaffold_benchmark_skill_rejects_missing_headings(tmp_path):
    report = tmp_path / "demo-report.md"
    report.write_text(
        "# Demo LakeQA Conversion Report\n\n"
        "This mentions ## Benchmark artifact inventory in prose only.\n",
        encoding="utf-8",
    )

    script = SCAFFOLDER / "scripts" / "scaffold_benchmark_skill.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(report),
            "--benchmark",
            "demo",
            "--output-root",
            str(tmp_path / "skills"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Report is missing required heading" in result.stderr


def test_scaffold_benchmark_skill_refuses_existing_destination(tmp_path):
    report = tmp_path / "demo-report.md"
    report.write_text(
        "# Demo LakeQA Conversion Report\n\n"
        "## Benchmark artifact inventory\n\n"
        "## Recommended benchmark-specific transform skill structure\n\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "skills"
    existing = output_root / "demo-lakeqa-transform"
    existing.mkdir(parents=True)

    script = SCAFFOLDER / "scripts" / "scaffold_benchmark_skill.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(report),
            "--benchmark",
            "demo",
            "--output-root",
            str(output_root),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Skill directory already exists" in result.stderr


def test_scaffold_benchmark_skill_generated_prompt_does_not_embed_report_answers(tmp_path):
    report = tmp_path / "demo-report.md"
    report.write_text(
        "# Demo LakeQA Conversion Report\n\n"
        "## Benchmark artifact inventory\n\n"
        "Example terminal answer: SECRET_FINAL_ANSWER\n\n"
        "## Recommended benchmark-specific transform skill structure\n\n"
        "Intermediate answer: SECRET_INTERMEDIATE_ANSWER\n\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "skills"

    script = SCAFFOLDER / "scripts" / "scaffold_benchmark_skill.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(report),
            "--benchmark",
            "demo",
            "--output-root",
            str(output_root),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    generated = (output_root / "demo-lakeqa-transform" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "SECRET_FINAL_ANSWER" not in generated
    assert "SECRET_INTERMEDIATE_ANSWER" not in generated
    assert "Do not copy answers" in generated
    assert "The report is a conversion-method artifact, not an answer key" in generated


def test_sana_profiling_ideal_artifact_skills_use_runtime_profile_convention():
    skill_names = ["author-ideal-profiles", "author-ideal-code", "profile-verifier"]

    for name in skill_names:
        skill = SANA_PROFILING / "skills" / name / "SKILL.md"
        assert skill.is_file()
        text = skill.read_text(encoding="utf-8")
        assert f"name: {name}" in text
        assert "runtime-profiles" in text
        assert "benchmarks/<benchmark>/tasks-mini/tasks" in text
        assert "runtime_profile_root" in text or "runtime-profile layout" in text

    ideal_profiles = (
        SANA_PROFILING / "skills" / "author-ideal-profiles" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "reasoning_chain_text" in ideal_profiles
    assert "no final answers" in ideal_profiles

    ideal_code = (
        SANA_PROFILING / "skills" / "author-ideal-code" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "ideal_code" in ideal_code
    assert "ideal_query" in ideal_code
    assert "HotpotQA-style text evidence tasks" in ideal_code

    verifier = (
        SANA_PROFILING / "skills" / "profile-verifier" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "Judge prompt leakage only in `reasoning_chain_text`" in verifier
    assert "metadata/audit fields" in verifier


def test_hotpotqa_generated_conversion_run_outputs_current_layout_and_no_prompt_leaks():
    run = RUNS / "hotpotqa-generated-conversion"
    validation = json.loads((run / "validation.json").read_text(encoding="utf-8"))

    assert validation["requested_import_count"] == 5
    assert validation["converted_count"] == 5
    assert validation["runtime_profile_count"] == 5
    assert validation["ideal_computation_result"] == "skipped_text_evidence_tasks"
    assert validation["prompt_surface_leaks"] == {}

    generated_skill = (
        run / "generated-skills" / "hotpotqa-lakeqa-transform" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "Do not copy answers" in generated_skill

    buckets = {row["bucket"] for row in validation["rows"]}
    assert buckets == {"k-1-d-2", "k-2-d-1"}
    for row in validation["rows"]:
        assert row["question_suffix_ok"] is True
        assert row["sources_rewritten"] is True
        assert row["subquestions_filled"] is True
        assert row["runtime_profile_uses_current_layout"] is True
        assert row["ideal_code_count"] == 0
        assert row["ideal_query_count"] == 0
        assert row["text_evidence_computation_skip_ok"] is True

        task = json.loads((ROOT / row["task"]).read_text(encoding="utf-8"))
        profile = json.loads((ROOT / row["runtime_profile"]).read_text(encoding="utf-8"))

        assert task["question"].endswith("Write your answer as [ANSWER].")
        assert all(
            not node["source"].startswith("hotpotqa://")
            for node in task["nodes"].values()
        )
        assert all(node["subquestion"] for node in task["nodes"].values())
        assert "runtime-profiles" in row["runtime_profile"]
        assert profile["ideal_code"] == []
        assert profile["ideal_query"] == []
        reasoning_text = "\n".join(profile["reasoning_chain_text"])
        assert task["answer"] not in reasoning_text
        assert task["answer"] not in generated_skill
