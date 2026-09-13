#!/usr/bin/env python3
"""Generate the complete experiment YAML grid programmatically.

Writes, into ``configs/generated/``:

* E1 main grid: 4 conditions x 2 mandated Instruct models x 3 seeds = 24 runs.
  Conditions: star_only, safety_replay, ewc_regularized, bianchi_control.
* E5 baselines: ewc_regularized at lambda in {50, 500, 5000} (1 seed, 1 model)
  and the Bianchi static control.

Every YAML is validated by round-tripping through ``conti.config_schema.load_config``
before it is written, so a schema-incompatible grid can never reach the cluster.
No hand-typed YAMLs anywhere: hand-editing per-run configs is how the old runs
silently drifted (see docs/plan.md NEVER list, item 4).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from conti.config_schema import load_config  # noqa: E402

MODELS = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen25": "Qwen/Qwen2.5-7B-Instruct",
}
E1_CONDITIONS = ("star_only", "safety_replay", "ewc_regularized", "bianchi_control")
E1_SEEDS = (42, 7, 123)
E5_LAMBDAS = (50.0, 500.0, 5000.0)


def base_config(model_id: str, condition: str, seed: int, output_dir: str, num_rounds: int) -> dict:
    return {
        "seed": seed,
        "output_dir": output_dir,
        "model": {
            "name_or_path": model_id,
            "torch_dtype": "bfloat16",
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "max_seq_length": 2048,
            "gradient_checkpointing": True,
        },
        "loop": {
            "experiment": condition,
            "num_self_improve_rounds": num_rounds,
            "generate_batch_size": 32,
            "train_micro_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2e-5,
            "num_epochs_per_round": 1,
            "max_train_samples_per_round": 2000,
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
            "generation_temperature": 0.7,
            "generation_top_p": 0.95,
            "generation_max_new_tokens": 512,
            "eval_every_round": True,
            "require_correct_trajectory": True,
            "max_grad_norm": 1.0,
            "probe_eval_size": 300,
            "full_eval_rounds": None,
            "eval_max_new_tokens": 512,
            "resume": True,
        },
        "data": {
            "math_dataset": "gsm8k",
            "math_split": "train",
            "math_test_split": "test",
            "safety_eval_datasets": ["advbench", "xstest", "donotanswer"],
            "max_prompts_per_eval": 1000,
        },
        "replay": {
            "enabled": condition in ("safety_replay", "bianchi_control"),
            "safety_samples_per_batch": 4,
            "dataset_path": None,
            "bianchi_safety_fraction": 0.03,
        },
        "ewc": {
            "enabled": condition == "ewc_regularized",
            "lambda_ewc": 5000.0,
            "fisher_samples": 256,
            "fisher_grad_clip": 1.0,
            "fisher_eps": 1e-8,
            "fisher_store_device": "cpu",
            "fisher_batch_size": 2,
            "fisher_max_len": 512,
        },
        "subliminal": {
            "tau_drift": 0.0,
            "anchor_samples": 32,
            "probe_samples": 32,
            "fisher_every_n_rounds": 2,
            "max_allowed_curvature": None,
            "cka_floor": None,
            "layer_indices": None,
            "probe_max_len": 256,
            "grad_clip": 1.0,
        },
        "verifier": {"mode": "keyword", "case_insensitive": True},
        "logging": {"use_wandb": False, "wandb_project": "conti-safety", "log_every_n_steps": 10},
    }


def build_grid(out_dir: Path, num_rounds: int) -> list[Path]:
    written: list[Path] = []

    def emit(name: str, payload: dict) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{name}.yaml"
        target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        load_config(target)  # round-trip validation; raises on schema drift
        written.append(target)
        print(f"wrote {target}")

    for model_key, model_id in MODELS.items():
        for condition in E1_CONDITIONS:
            for seed in E1_SEEDS:
                name = f"e1_{condition}_{model_key}_s{seed}"
                payload = base_config(
                    model_id,
                    condition,
                    seed,
                    f"outputs/e1/{condition}/{model_key}/{seed}",
                    num_rounds,
                )
                emit(name, payload)

    llama_id = MODELS["llama3"]
    for lambda_ewc in E5_LAMBDAS:
        payload = base_config(
            llama_id, "ewc_regularized", 42, f"outputs/e5/ewc_l{int(lambda_ewc)}/llama3/42", num_rounds
        )
        payload["ewc"]["lambda_ewc"] = lambda_ewc
        emit(f"e5_ewc_l{int(lambda_ewc)}", payload)

    payload = base_config(
        llama_id, "bianchi_control", 42, "outputs/e5/bianchi_static/llama3/42", num_rounds
    )
    payload["replay"]["bianchi_safety_fraction"] = 0.03
    emit("e5_bianchi_static", payload)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: configs/generated)")
    parser.add_argument("--num-rounds", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "configs" / "generated"
    files = build_grid(out_dir, args.num_rounds)
    print(f"\n{len(files)} configs generated and schema-validated in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
