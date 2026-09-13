"""Tests for the judge, grid generator, resume logic, calibration and rollup.

Complements tests/test_subliminal_suite.py: these cover the operational layer
(scripts + config plumbing) with zero network access and zero GPU requirements.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from conti.eval.judge import GuardParseError, export_spot_check_csv, parse_guard_output  # noqa: E402
from conti.loop.run import _find_resume_round, _normalize_benchmark_names  # noqa: E402
from scripts.calibrate_filter import choose_threshold, roc_auc  # noqa: E402
from scripts.make_grid import E1_CONDITIONS, MODELS, base_config, build_grid  # noqa: E402
from scripts.stats_rollup import _dig, _final_full_row, collect_runs  # noqa: E402


# --------------------------------------------------------------------------- #
# Llama-Guard output parsing
# --------------------------------------------------------------------------- #


class TestGuardParsing:
    def test_safe(self) -> None:
        label, category = parse_guard_output("safe")
        assert label == "safe"
        assert category is None

    def test_safe_with_padding(self) -> None:
        label, _ = parse_guard_output("  safe\n")
        assert label == "safe"

    def test_unsafe_with_category(self) -> None:
        label, category = parse_guard_output("unsafe\nS10")
        assert label == "unsafe"
        assert category == "S10"

    def test_unsafe_without_category(self) -> None:
        label, category = parse_guard_output("unsafe")
        assert label == "unsafe"
        assert category is None

    def test_unsafe_with_oui_category(self) -> None:
        label, category = parse_guard_output("unsafe\nO3")
        assert label == "unsafe"
        assert category == "O3"

    @pytest.mark.parametrize("raw", ["", None, "maybe", "i cannot parse this"])
    def test_unparseable_raises(self, raw) -> None:
        with pytest.raises(GuardParseError):
            parse_guard_output(raw)


class TestSpotCheckCSV:
    def test_writes_aligned_rows(self, tmp_path: Path) -> None:
        target = tmp_path / "spot.csv"
        count = export_spot_check_csv(
            str(target),
            ["p1", "p2"],
            ["r1", "r2"],
            [True, False],
            model_id="test-judge",
        )
        assert count == 2
        lines = target.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        assert "test-judge" in lines[1]

    def test_length_mismatch_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            export_spot_check_csv(str(tmp_path / "s.csv"), ["p"], [], [True])


# --------------------------------------------------------------------------- #
# resume + benchmark normalization (conti.loop.run helpers)
# --------------------------------------------------------------------------- #


class TestResumeRoundScan:
    def test_empty_dir_returns_minus_one(self, tmp_path: Path) -> None:
        assert _find_resume_round(tmp_path) == -1

    def test_highest_done_sentinel_wins(self, tmp_path: Path) -> None:
        for round_index in (0, 1, 2):
            ckpt = tmp_path / "checkpoints" / f"round_{round_index}"
            ckpt.mkdir(parents=True)
            (ckpt / ".done").write_text("ok\n")
        assert _find_resume_round(tmp_path) == 2

    def test_crashed_save_is_ignored(self, tmp_path: Path) -> None:
        done_ckpt = tmp_path / "checkpoints" / "round_1"
        done_ckpt.mkdir(parents=True)
        (done_ckpt / ".done").write_text("ok\n")
        crashed = tmp_path / "checkpoints" / "round_2"
        crashed.mkdir(parents=True)
        (crashed / "adapter_model.safetensors").write_text("partial")
        assert _find_resume_round(tmp_path) == 1

    def test_non_numeric_dirs_ignored(self, tmp_path: Path) -> None:
        junk = tmp_path / "checkpoints" / "round_latest"
        junk.mkdir(parents=True)
        (junk / ".done").write_text("ok\n")
        assert _find_resume_round(tmp_path) == -1


class TestBenchmarkNormalization:
    def test_strips_legacy_suffixes(self) -> None:
        assert _normalize_benchmark_names(
            ["advbench_subset", "xstest_full", "donotanswer"]
        ) == ("advbench", "xstest", "donotanswer")

    def test_deduplicates_and_preserves_order(self) -> None:
        assert _normalize_benchmark_names(["advbench", "adv_bench", "advbench"]) == ("advbench",)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError):
            _normalize_benchmark_names(["toxicity"])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            _normalize_benchmark_names([])


# --------------------------------------------------------------------------- #
# calibration math helpers
# --------------------------------------------------------------------------- #


class TestChooseThreshold:
    def test_ninety_percent_pass_rate(self) -> None:
        cosines = [0.0] * 9 + [1.0]
        tau = choose_threshold(cosines, min_pass_rate=0.9)
        assert tau == pytest.approx(0.0)
        passing = sum(1 for value in cosines if value >= tau)
        assert passing / len(cosines) >= 0.9

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            choose_threshold([], 0.9)

    def test_invalid_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            choose_threshold([0.5], 0.0)

    def test_full_pass_rate_takes_minimum(self) -> None:
        values = [0.2, 0.5, 0.8]
        assert choose_threshold(values, min_pass_rate=1.0) == pytest.approx(0.2)


class TestRocAuc:
    def test_perfect_separation(self) -> None:
        assert roc_auc([0.8, 0.9, 1.0], [0.1, 0.2, 0.3]) == pytest.approx(1.0)

    def test_reversed_separation(self) -> None:
        assert roc_auc([0.1, 0.2], [0.8, 0.9]) == pytest.approx(0.0)

    def test_identical_distributions(self) -> None:
        assert roc_auc([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)

    def test_empty_side_raises(self) -> None:
        with pytest.raises(ValueError):
            roc_auc([], [0.5])


# --------------------------------------------------------------------------- #
# grid generator
# --------------------------------------------------------------------------- #


class TestGridGenerator:
    def test_base_config_roundtrips_through_schema(self, tmp_path: Path) -> None:
        from conti.config_schema import load_config

        payload = base_config(MODELS["llama3"], "safety_replay", 7, "outputs/x", 10)
        target = tmp_path / "probe.yaml"
        target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        cfg = load_config(target)
        assert cfg.loop.experiment == "safety_replay"
        assert cfg.model.name_or_path == MODELS["llama3"]
        assert cfg.replay.enabled is True
        assert cfg.ewc.enabled is False
        assert cfg.loop.resume is True

    def test_condition_flag_consistency(self) -> None:
        for condition in ("star_only", "ewc_regularized", "bianchi_control"):
            payload = base_config(MODELS["qwen25"], condition, 42, "outputs/x", 10)
            assert payload["replay"]["enabled"] == (condition in ("safety_replay", "bianchi_control"))
            assert payload["ewc"]["enabled"] == (condition == "ewc_regularized")

    def test_build_grid_writes_and_validates(self, tmp_path: Path) -> None:
        files = build_grid(tmp_path, num_rounds=10)
        expected_e1 = 2 * len(E1_CONDITIONS) * 3
        assert len(files) == expected_e1 + 4  # 3 EWC lambdas + bianchi static
        names = {path.name for path in files}
        for condition in E1_CONDITIONS:
            for model_key in MODELS:
                for seed in (42, 7, 123):
                    assert f"e1_{condition}_{model_key}_s{seed}.yaml" in names
        assert "e5_ewc_l5000.yaml" in names
        assert "e5_bianchi_static.yaml" in names

    def test_ewc_lambda_sweep_values(self, tmp_path: Path) -> None:
        from conti.config_schema import load_config

        build_grid(tmp_path, num_rounds=10)
        cfg = load_config(tmp_path / "e5_ewc_l500.yaml")
        assert cfg.ewc.lambda_ewc == 500.0
        assert cfg.loop.experiment == "ewc_regularized"


# --------------------------------------------------------------------------- #
# stats rollup
# --------------------------------------------------------------------------- #


def _write_run(
    root: Path,
    run_name: str,
    condition: str,
    model: str,
    seed: int,
    final_asr: float,
) -> Path:
    run_dir = root / run_name
    (run_dir / "logs").mkdir(parents=True)
    manifest = {"seed": seed, "config_dict": {"model": {"name_or_path": model}}}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    probe_row = {"round": 4, "condition": condition, "seed": seed, "eval_mode": "probe",
                 "safety": {"aggregate_unsafe": {"asr": 0.9}}}
    final_row = {
        "round": 9,
        "condition": condition,
        "seed": seed,
        "eval_mode": "final",
        "accepted": 40,
        "rej_gate_r": 3,
        "math_pass1": {"accuracy": 0.5},
        "safety": {
            "advbench": {"unsafe": {"asr": final_asr}},
            "xstest": {"unsafe": {"asr": final_asr + 0.1}, "safe": {"false_refusal_rate": 0.2}},
            "donotanswer": {"unsafe": {"asr": final_asr + 0.2}},
            "aggregate_unsafe": {"asr": final_asr + 0.05},
        },
    }
    with (run_dir / "logs" / "round_metrics.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(probe_row) + "\n")
        handle.write(json.dumps(final_row) + "\n")
    return run_dir


class TestStatsRollup:
    def test_dig_paths(self) -> None:
        payload = {"a": {"b": {"c": 3.5}}}
        assert _dig(payload, ("a", "b", "c")) == 3.5
        assert _dig(payload, ("a", "missing")) is None
        assert _dig(payload, ("a", "b", "c", "d")) is None

    def test_final_full_row_prefers_final(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path, "run_a", "star_only", "m", 42, 0.3)
        row = _final_full_row(run_dir)
        assert row is not None
        assert row["eval_mode"] == "final"
        assert row["round"] == 9

    def test_collect_runs_and_summary(self, tmp_path: Path) -> None:
        _write_run(tmp_path, "run_a", "star_only", "model-x", 42, 0.3)
        _write_run(tmp_path, "run_b", "star_only", "model-x", 7, 0.5)
        records = collect_runs(tmp_path)
        assert len(records) == 2
        by_seed = {record["seed"]: record for record in records}
        assert by_seed[42]["aggregate_asr"] == pytest.approx(0.35)
        assert by_seed[7]["xstest_safe_frr"] == pytest.approx(0.2)

        from scripts.stats_rollup import write_rollup_csv, write_summary_md

        write_rollup_csv(records, tmp_path / "rollup.csv")
        write_summary_md(records, tmp_path / "summary.md")
        csv_text = (tmp_path / "rollup.csv").read_text(encoding="utf-8")
        assert "star_only" in csv_text and "model-x" in csv_text
        md_text = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "0.4000 +/-" in md_text  # mean of 0.3 and 0.5

    def test_incomplete_run_skipped(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_c"
        run_dir.mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
        (run_dir / "logs").mkdir()
        (run_dir / "logs" / "round_metrics.jsonl").write_text("", encoding="utf-8")
        assert collect_runs(tmp_path) == []


# --------------------------------------------------------------------------- #
# build_probes hashing helper (offline; no dataset download)
# --------------------------------------------------------------------------- #


class TestBuildProbesHelpers:
    def _load_module(self):
        spec = importlib.util.spec_from_file_location(
            "build_probes", REPO_ROOT / "scripts" / "build_probes.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module

    def test_sha256_is_stable(self, tmp_path: Path) -> None:
        module = self._load_module()
        target = tmp_path / "blob.bin"
        target.write_bytes(b"continual")
        first = module.sha256_of(target)
        second = module.sha256_of(target)
        assert first == second
        assert len(first) == 64
        target.write_bytes(b"continual!")
        assert module.sha256_of(target) != first
