"""Validation-only task-specific ensemble evaluation for Saffaryazdi runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config.config import Config
from src.models.saffaryazdi_lstm import SaffaryazdiTaskModel
from src.pipeline.saffaryazdi_pipeline import SaffaryazdiPipeline
from train_saffaryazdi_lstm import (
    choose_fusion_weight,
    make_loader,
    metrics,
    predict,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--checkpoint-dir", default="outputs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="thesis-emotion-recognition")
    parser.add_argument("--wandb-run-name", default="saffaryazdi-task-ensemble-search")
    parser.add_argument("--selection", choices=("macro_f1", "subject_g"), default="macro_f1")
    return parser.parse_args()


def load_candidate(path, arrays, task, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    model = SaffaryazdiTaskModel(
        input_dim=arrays["train"]["eeg"].shape[-1],
        phys_dim=arrays["train"]["phys"].shape[-1],
        hidden1=args["hidden1"],
        hidden2=args["hidden2"],
        dropout=args["dropout"],
        pooling=args["pooling"],
        fusion_mode=args["fusion"],
        bidirectional=args.get("bidirectional", False),
        temporal_encoder=args.get("temporal_encoder", "lstm"),
    ).to(device)
    model.load_state_dict(checkpoint["state_dicts"][task])
    val_loader = make_loader(arrays["validation"], task, 32, False)
    test_loader = make_loader(arrays["test"], task, 32, False)
    val = predict(model, val_loader, device, args["fusion"])
    test = predict(model, test_loader, device, args["fusion"])
    return args, val, test


def candidate_probabilities(args, prediction, labels, subjects):
    eeg, phys, learned, _ = prediction
    if args["fusion"] != "late":
        return learned
    alpha, _, _ = choose_fusion_weight(eeg, phys, labels, subjects, "subject_g")
    return alpha * eeg + (1.0 - alpha) * phys


def main():
    args = parse_args()
    device = torch.device(args.device)
    config = Config()
    pipeline = SaffaryazdiPipeline(config, seed=42)
    arrays = pipeline.run(cache_path=args.cache_path)
    subjects = {
        split: arrays[split]["subjects"].astype(str)
        for split in ("validation", "test")
    }
    labels = {
        task: arrays["validation"][task] for task in ("valence", "arousal")
    }
    candidates = {
        "base80_subjectg": "model_search_base_subjectg_seed42.pt",
        "regularized48": "saffaryazdi_final_regularized_48_16.pt",
        "macro48": "model_search_macro_48_16_seed42.pt",
        "gated48": "attention_gated_seed42.pt",
        "learned48": "attention_learned_seed42.pt",
        "crossmodal48": "attention_crossmodal_seed42.pt",
    }
    loaded = {task: {} for task in ("valence", "arousal")}
    for name, filename in candidates.items():
        path = Path(args.checkpoint_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        for task in loaded:
            loaded[task][name] = load_candidate(path, arrays, task, device)

    results = {"candidates": {}, "tasks": {}}
    for task in ("valence", "arousal"):
        val_probabilities = {}
        test_probabilities = {}
        val_labels = arrays["validation"][task]
        test_labels = arrays["test"][task]
        for name, (model_args, val_prediction, test_prediction) in loaded[task].items():
            val_probabilities[name] = candidate_probabilities(
                model_args, val_prediction, val_labels, subjects["validation"]
            )
            test_probabilities[name] = candidate_probabilities(
                model_args, test_prediction, test_labels, subjects["test"]
            )
            results["candidates"].setdefault(name, {})[task] = {
                "validation": metrics(
                    val_probabilities[name].argmax(1),
                    val_labels,
                    subjects["validation"],
                ),
                "test": metrics(
                    test_probabilities[name].argmax(1),
                    test_labels,
                    subjects["test"],
                ),
            }

        names = list(val_probabilities)
        combinations = []
        for name in names:
            combinations.append((name, val_probabilities[name], test_probabilities[name]))
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                best = None
                for weight in np.linspace(0.0, 1.0, 21):
                    val_prob = weight * val_probabilities[left] + (1 - weight) * val_probabilities[right]
                    val_metrics = metrics(val_prob.argmax(1), val_labels, subjects["validation"])
                    if args.selection == "subject_g":
                        score = (
                            val_metrics["subject_f1_mean"],
                            val_metrics["worst_subject_f1"],
                            -val_metrics["subject_f1_std"],
                            val_metrics["macro_f1"],
                        )
                    else:
                        score = (
                            val_metrics["macro_f1"],
                            val_metrics["subject_f1_mean"],
                            -val_metrics["subject_f1_std"],
                        )
                    if best is None or score > best[0]:
                        test_prob = weight * test_probabilities[left] + (1 - weight) * test_probabilities[right]
                        best = (score, f"{left}+{right}", weight, val_metrics, val_prob, test_prob)
                combinations.append((best[1], best[4], best[5]))
                results["tasks"].setdefault(task, {}).setdefault("pair_search", []).append(
                    {
                        "models": [left, right],
                        "left_weight": best[2],
                        "validation": best[3],
                        "test": metrics(best[5].argmax(1), test_labels, subjects["test"]),
                    }
                )
        all_prob = np.mean([val_probabilities[name] for name in names], axis=0)
        all_test = np.mean([test_probabilities[name] for name in names], axis=0)
        combinations.append(("all_equal", all_prob, all_test))
        best_name, best_val, best_test = max(
            combinations,
            key=lambda item: (
                (
                    metrics(item[1].argmax(1), val_labels, subjects["validation"])["subject_f1_mean"],
                    metrics(item[1].argmax(1), val_labels, subjects["validation"])["worst_subject_f1"],
                    -metrics(item[1].argmax(1), val_labels, subjects["validation"])["subject_f1_std"],
                    metrics(item[1].argmax(1), val_labels, subjects["validation"])["macro_f1"],
                )
                if item[1] is not None and args.selection == "subject_g"
                else (
                    metrics(item[1].argmax(1), val_labels, subjects["validation"])["macro_f1"],
                    metrics(item[1].argmax(1), val_labels, subjects["validation"])["subject_f1_mean"],
                    -metrics(item[1].argmax(1), val_labels, subjects["validation"])["subject_f1_std"],
                )
                if item[1] is not None
                else (-1.0,),
            ),
        )
        if best_name == "all_equal":
            best_val = all_prob
        best_val_metrics = metrics(best_val.argmax(1), val_labels, subjects["validation"])
        best_test_metrics = metrics(best_test.argmax(1), test_labels, subjects["test"])
        results["tasks"][task]["selected"] = {
            "name": best_name,
            "validation": best_val_metrics,
            "test": best_test_metrics,
        }

    results["summary"] = {
        "mean_test_f1": float(
            np.mean(
                [
                    results["tasks"][task]["selected"]["test"]["macro_f1"]
                    for task in ("valence", "arousal")
                ]
            )
        )
    }
    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "selection": "validation_only",
                "candidate_count": len(candidates),
                "cache_path": args.cache_path,
            },
        )
        for task in ("valence", "arousal"):
            selected = results["tasks"][task]["selected"]
            wandb.log(
                {
                    f"{task}/validation_macro_f1": selected["validation"]["macro_f1"],
                    f"{task}/test_macro_f1": selected["test"]["macro_f1"],
                    f"{task}/test_accuracy": selected["test"]["accuracy"],
                }
            )
        wandb.log({"ensemble/mean_test_f1": results["summary"]["mean_test_f1"]})
        run.summary["selected_valence"] = results["tasks"]["valence"]["selected"]["name"]
        run.summary["selected_arousal"] = results["tasks"]["arousal"]["selected"]["name"]
        run.finish()
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
