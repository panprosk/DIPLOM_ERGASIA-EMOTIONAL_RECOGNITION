"""
optuna_search.py

Αυτοματοποιημένη αναζήτηση υπερπαραμέτρων (hyperparameter search) με
Optuna, για ΚΑΙ τα δύο μοντέλα της διπλωματικής (Hybrid CNN-MLP,
Cross-Attention Transformer).

Γιατί δεν χρησιμοποιούμε απλό validation accuracy ως objective
--------------------------------------------------------------
Ζητούμενο ρητά: το objective να είναι το "Cross-Subject Generalization
Score" (βλ. src/utils/generalization_score.py):

    G = 0.5 * mean_subject_f1 + 0.3 * worst_subject_f1 - 0.2 * subject_f1_std

δηλαδή ανταμείβει μοντέλα που (α) έχουν καλό μέσο όρο macro-F1 ανά
subject, (β) δεν "καταρρέουν" σε κάποιο συγκεκριμένο subject (worst-
case), και (γ) έχουν όσο πιο ομοιόμορφη απόδοση γίνεται ανάμεσα σε
διαφορετικά subjects (χαμηλή διακύμανση) -- ακριβώς το πρόβλημα της
cross-subject γενίκευσης στο οποίο επικεντρώνεται η διπλωματική.

Πώς λειτουργεί
--------------
1. Το data pipeline (φόρτωση/προεπεξεργασία/feature extraction/
   windowing/datasets) τρέχει ΜΙΑ ΦΟΡΑ στην αρχή -- ΟΧΙ ανά trial.
   Αυτό είναι κρίσιμο για να είναι εφικτό το search σε CPU-only
   μηχάνημα (το ίδιο το pipeline παίρνει ~1-2+ ώρες).
2. Για κάθε Optuna trial: δειγματοληπτούνται υπερπαράμετροι
   (learning rate, weight decay, dropout, batch size, adversarial/
   contrastive weights, αρχιτεκτονικές παράμετροι κ.λπ.), χτίζεται
   ΝΕΟ μοντέλο, και εκπαιδεύεται για λίγα epochs (--epochs-per-trial).
3. Μετά από ΚΑΘΕ epoch, υπολογίζεται το Generalization Score στο
   validation set και αναφέρεται στο Optuna (`trial.report`) -- αν
   το trial φαίνεται να πηγαίνει σαφώς χειρότερα από τα μέχρι τώρα
   καλύτερα trials, "κόβεται" νωρίς (pruning, MedianPruner) ώστε να
   μη σπαταληθεί χρόνος σε trials που δεν έχουν ελπίδα.
4. Όλα τα trials (παράμετροι, ενδιάμεσα scores, αποτέλεσμα) γράφονται
   σε ΜΟΝΙΜΗ SQLite βάση (Optuna storage) -- το study μπορεί να
   ξανατρέξει/συνεχιστεί οποτεδήποτε με το ΙΔΙΟ study-name.
5. Στο τέλος τυπώνονται/αποθηκεύονται τα καλύτερα hyperparameters
   (JSON) + το checkpoint του καλύτερου μοντέλου που βρέθηκε συνολικά
   + optional HTML plots (optimization history, param importance).



"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn

import train_hybrid_baseline as hybrid_module
import train_cross_attention_transformer as transformer_module

from src.config.config import Config
from src.dataloaders.dataloader_factory import DataLoaderFactory
from src.pipeline.data_pipeline import DataPipeline
from src.models import build_model
from src.models.supervised_contrastive_loss import SupervisedContrastiveLoss
from src.utils.generalization_score import compute_generalization_score


def parse_args():

    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search (Cross-Subject Generalization Score)."
    )

    parser.add_argument("--model", choices=["hybrid", "cross_attention_transformer"],
                         required=True, help="Ποιο μοντέλο να βελτιστοποιηθεί.")

    parser.add_argument("--n-trials", type=int, default=40,
                         help="Πόσα Optuna trials να τρέξουν (default: 40).")
    parser.add_argument("--timeout", type=int, default=None,
                         help="Μέγιστος συνολικός χρόνος σε δευτερόλεπτα "
                              "(π.χ. 14400 = 4 ώρες). Default: κανένα όριο, "
                              "σταματάει μόνο όταν ολοκληρωθούν τα --n-trials.")
    parser.add_argument("--epochs-per-trial", type=int, default=10,
                         help="Μέγιστα epochs ανά trial (default: 10).")
    parser.add_argument("--trial-patience", type=int, default=4,
                         help="Early stopping ΜΕΣΑ σε κάθε trial: σταματάει "
                              "νωρίτερα αν δεν βελτιωθεί το Generalization "
                              "Score για τόσα epochs (default: 4).")

    parser.add_argument("--num-subjects", type=int, default=None,
                         help="Χρησιμοποιεί μόνο τα πρώτα N subjects (sanity "
                              "check ή γρηγορότερο search σε subset). Default: όλα (32).")
    parser.add_argument("--trials-per-subject", type=int, default=None,
                         help="Κόβει κάθε subject στα πρώτα N trials. Default: όλα (40).")

    parser.add_argument("--study-name", type=str, default=None,
                         help="Optuna study name (default: '<model>_cross_subject_generalization'). "
                              "Ξανατρέχοντας με το ΙΔΙΟ όνομα/storage, το study συνεχίζεται "
                              "(τα προηγούμενα trials ΔΕΝ χάνονται).")
    parser.add_argument("--storage", type=str, default=None,
                         help="Optuna storage URL (default: sqlite:///outputs/optuna_studies.db).")
    parser.add_argument("--cache-path", type=str, default=None,
                         help="Pickle cache των datasets. Αν υπάρχει, "
                              "παρακάμπτεται το CPU-bound preprocessing.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true",
                         help="Απενεργοποιεί το EEG data augmentation στο search.")
    parser.add_argument("--search-profile", choices=[
        "focused", "refine", "best_recipe", "cross_subject_v2"
    ],
                        default="focused",
                        help="focused: στενό, evidence-based Hybrid search.")
    parser.add_argument("--objective", choices=["generalization", "trial_f1"],
                        default="generalization",
                        help="Optuna objective. trial_f1 matches the training "
                            "checkpoint metric used by the best historical run.")

    parser.add_argument("--w-mean", type=float, default=0.5,
                         help="Βάρος του mean_subject_f1 στο Generalization Score.")
    parser.add_argument("--w-worst", type=float, default=0.3,
                         help="Βάρος του worst_subject_f1 στο Generalization Score.")
    parser.add_argument("--w-var", type=float, default=0.2,
                         help="Βάρος (penalty) του subject_f1_std στο Generalization Score.")

    return parser.parse_args()


# Search spaces ανά μοντέλο. Κρατημένα σκόπιμα γύρω από τιμές που ήδη
# ξέρουμε ότι δουλεύουν λογικά καλά (βλ. ΚΑΤΑΓΡΑΦΗ_ΑΛΛΑΓΩΝ...txt) ώστε
# το Optuna να μην σπαταλά trials σε προφανώς άσχημες περιοχές.

def _sample_common_hparams(trial, config, profile="focused"):

    # Ranges narrowed γύρω από την περιοχή σύγκλισης των Top-5 trials
    # του hybrid_v2 study (βλ. optuna_best_params_hybrid.json) -- τα
    # καλύτερα trials είχαν όλα batch_size=128, embedding_dim>=96, και
    # label_smoothing χαμηλό. Στόχος: λιγότερα trials σπαταλημένα σε
    # ήδη αποδεδειγμένα χειρότερες περιοχές (bs=32/48, embedding_dim=64).
    if profile == "best_recipe" and config.MODEL_NAME == "hybrid":
        config.LEARNING_RATE = trial.suggest_float("lr", 1.5e-4, 4.5e-4, log=True)
        config.WEIGHT_DECAY = trial.suggest_float("weight_decay", 5e-4, 1.5e-3, log=True)
        config.DROPOUT = trial.suggest_float("dropout", 0.25, 0.40)
        config.BATCH_SIZE = trial.suggest_categorical("batch_size", [64, 96, 128])
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.06)
        grad_clip = trial.suggest_float("grad_clip", 0.7, 1.2)
        config.ADVERSARIAL_LAMBDA_MAX = trial.suggest_float(
            "adversarial_lambda_max", 0.24, 0.32
        )
        adversarial_weight = trial.suggest_float("adversarial_weight", 0.24, 0.36)
        contrastive_weight = trial.suggest_float("contrastive_weight", 0.14, 0.24)
        contrastive_temperature = trial.suggest_float(
            "contrastive_temperature", 0.05, 0.08
        )
        gate_balance_weight = 0.0
    elif profile == "cross_subject_v2" and config.MODEL_NAME == "hybrid":
        # Evidence-based region from the historical Hybrid run, while
        # explicitly searching the fusion imbalance observed in the latest
        # confirmation run (EEG gate ~= 0.80).
        config.LEARNING_RATE = trial.suggest_float("lr", 1.5e-4, 4.5e-4, log=True)
        config.WEIGHT_DECAY = trial.suggest_float("weight_decay", 5e-4, 2e-3, log=True)
        config.DROPOUT = trial.suggest_float("dropout", 0.30, 0.50)
        config.BATCH_SIZE = trial.suggest_categorical("batch_size", [48, 64, 96])
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.08)
        grad_clip = trial.suggest_float("grad_clip", 0.8, 1.3)
        config.ADVERSARIAL_LAMBDA_MAX = trial.suggest_float(
            "adversarial_lambda_max", 0.15, 0.35
        )
        adversarial_weight = trial.suggest_float("adversarial_weight", 0.15, 0.35)
        contrastive_weight = trial.suggest_float("contrastive_weight", 0.05, 0.25)
        contrastive_temperature = trial.suggest_float(
            "contrastive_temperature", 0.06, 0.09
        )
        gate_balance_weight = trial.suggest_float("gate_balance_weight", 0.0, 0.12)
    elif profile == "refine" and config.MODEL_NAME == "hybrid":
        config.LEARNING_RATE = trial.suggest_float("lr", 3.5e-4, 7.5e-4, log=True)
        config.WEIGHT_DECAY = trial.suggest_float("weight_decay", 2e-4, 8e-4, log=True)
        config.DROPOUT = trial.suggest_float("dropout", 0.40, 0.50)
        config.BATCH_SIZE = 128
        label_smoothing = trial.suggest_float("label_smoothing", 0.04, 0.08)
        grad_clip = trial.suggest_float("grad_clip", 1.0, 1.5)
        config.ADVERSARIAL_LAMBDA_MAX = trial.suggest_float(
            "adversarial_lambda_max", 0.22, 0.34
        )
        adversarial_weight = trial.suggest_float("adversarial_weight", 0.28, 0.42)
        contrastive_weight = trial.suggest_float("contrastive_weight", 0.16, 0.30)
        contrastive_temperature = trial.suggest_float(
            "contrastive_temperature", 0.065, 0.095
        )
    else:
        config.LEARNING_RATE = trial.suggest_float("lr", 2e-4, 8e-4, log=True)
        config.WEIGHT_DECAY = trial.suggest_float("weight_decay", 3e-4, 2e-3, log=True)
        config.DROPOUT = trial.suggest_float("dropout", 0.30, 0.50)
        config.BATCH_SIZE = trial.suggest_categorical("batch_size", [96, 128])
        label_smoothing = trial.suggest_float("label_smoothing", 0.02, 0.08)
        grad_clip = trial.suggest_float("grad_clip", 0.8, 1.4)
        config.ADVERSARIAL_LAMBDA_MAX = trial.suggest_float(
            "adversarial_lambda_max", 0.15, 0.35
        )
        adversarial_weight = trial.suggest_float("adversarial_weight", 0.20, 0.40)
        contrastive_weight = trial.suggest_float("contrastive_weight", 0.10, 0.30)
        contrastive_temperature = trial.suggest_float(
            "contrastive_temperature", 0.06, 0.10
        )

    return {
        "label_smoothing": label_smoothing,
        "grad_clip": grad_clip,
        "adversarial_weight": adversarial_weight,
        "contrastive_weight": contrastive_weight,
        "contrastive_temperature": contrastive_temperature,
        "gate_balance_weight": gate_balance_weight,
    }


def _sample_hybrid_hparams(trial, config, profile="focused"):

    choices = [64, 96, 128] if profile == "cross_subject_v2" else (
        [96, 128] if profile == "best_recipe" else (
        [128, 160] if profile == "refine" else [96, 128, 160]
        )
    )
    config.EMBEDDING_DIM = trial.suggest_categorical("embedding_dim", choices)


def _sample_transformer_hparams(trial, config):

    # Refinement γύρω από την περιοχή σύγκλισης του Top-5 του πρώτου
    # transformer_v1 study (42 trials, βλ. optuna_history/importance
    # HTML): ΟΛΑ τα Top-5 trials είχαν patch_size=32 και d_model=32 σε
    # 4/5 -- κλειδώνουμε αυτά, στενεύουμε τα υπόλοιπα γύρω από το
    # παρατηρημένο εύρος, ώστε τα νέα trials να μην ξοδεύονται σε ήδη
    # αποδεδειγμένα χειρότερες περιοχές του search space.
    config.TRANSFORMER_D_MODEL = trial.suggest_categorical("d_model", [32, 64])
    config.TRANSFORMER_NUM_HEADS = trial.suggest_categorical("num_heads", [2, 4])
    config.TRANSFORMER_SELF_ATTN_LAYERS = trial.suggest_int("self_attn_layers", 2, 4)
    config.TRANSFORMER_CROSS_ATTN_LAYERS = trial.suggest_int("cross_attn_layers", 1, 2)
    config.TRANSFORMER_PATCH_SIZE = trial.suggest_categorical("patch_size", [32, 48])

    ff_mult = trial.suggest_categorical("ff_mult", [2, 4])
    config.TRANSFORMER_FF_DIM = ff_mult * config.TRANSFORMER_D_MODEL

    # Ξεχωριστό dropout config field (βλ. σχόλιο στο config.py) -- το
    # μοντέλο διαβάζει config.TRANSFORMER_DROPOUT, όχι config.DROPOUT.
    config.TRANSFORMER_DROPOUT = config.DROPOUT

    # Target-entropy regularizer πάνω στο cross-attention (βλ.
    # train_cross_attention_transformer.py). Εύρος γύρω από το
    # production default (0.05) που επιβεβαιώθηκε να δίνει υγιές
    # (όχι πλήρως uniform ούτε πλήρως collapsed) attention pattern.
    attn_entropy_weight = trial.suggest_float("attn_entropy_weight", 0.02, 0.10)

    return {"attn_entropy_weight": attn_entropy_weight}


class Objective:
    """
    Callable Optuna objective. Κρατάει state ΑΝΑΜΕΣΑ σε trials (μέσω
    instance attributes, όχι μέσω global variables) ώστε να θυμάται
    το ΚΑΛΥΤΕΡΟ μοντέλο/παραμέτρους σε ΟΛΟ το study (όχι μόνο μέσα σε
    ένα trial), και να το αποθηκεύει σε checkpoint αμέσως μόλις
    βρεθεί καλύτερο -- έτσι δεν χάνεται τίποτα αν διακοπεί το script.
    """

    def __init__(self, args, base_config, module, device,
                 train_dataset, validation_dataset, test_dataset,
                 train_labels, subject_to_idx, num_subjects, output_dir):

        self.args = args
        self.base_config = base_config
        self.module = module
        self.device = device
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.train_labels = train_labels
        self.subject_to_idx = subject_to_idx
        self.num_subjects = num_subjects
        self.output_dir = output_dir

        self.best_score = -float("inf")
        self.best_params = None
        self.best_val_metrics = None
        self.best_trial_number = None

    def __call__(self, trial):

        config = copy.deepcopy(self.base_config)

        extra = _sample_common_hparams(
            trial, config, profile=self.args.search_profile
        )

        if self.args.model == "hybrid":
            _sample_hybrid_hparams(
                trial, config, profile=self.args.search_profile
            )
        else:
            extra.update(_sample_transformer_hparams(trial, config))

        # Ξαναχτίζει ΜΟΝΟ τα DataLoaders (φθηνό -- ήδη επεξεργασμένα
        # in-memory datasets) με το batch size αυτού του trial, ΧΩΡΙΣ
        # να ξανατρέξει ολόκληρο το data pipeline.
        loader_factory = DataLoaderFactory(
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            weighted_sampling=getattr(config, "WEIGHTED_SAMPLING", True),
        )
        loaders = loader_factory.build(
            train_dataset=self.train_dataset,
            val_dataset=self.validation_dataset,
            test_dataset=self.test_dataset,
            train_labels=self.train_labels,
        )
        train_loader = loaders["train"]
        validation_loader = loaders["validation"]

        torch.manual_seed(self.args.seed + trial.number)

        model = build_model(
            config.MODEL_NAME, config, num_subjects=self.num_subjects
        ).to(self.device)

        criterion = nn.CrossEntropyLoss(label_smoothing=extra["label_smoothing"])

        optimizer_cls = torch.optim.AdamW if self.args.model == "hybrid" else torch.optim.Adam
        optimizer = optimizer_cls(
            model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
        )

        contrastive_loss_fn = SupervisedContrastiveLoss(
            temperature=extra["contrastive_temperature"]
        )

        best_trial_score = -float("inf")
        best_trial_state = None
        best_trial_val_metrics = None
        gen_score_history = []
        epochs_no_improve = 0

        print(
            f"\n--- Trial {trial.number} ---\nParams: {trial.params}\n",
            flush=True,
        )

        for epoch in range(1, self.args.epochs_per_trial + 1):

            # Ίδιο DANN-style linear ramp-up με τα κανονικά training scripts.
            progress = min(1.0, (epoch - 1) / max(1, self.args.epochs_per_trial / 2))
            grl_lambda = config.ADVERSARIAL_LAMBDA_MAX * progress

            run_epoch_kwargs = dict(
                adversarial_weight=extra["adversarial_weight"],
                contrastive_loss_fn=contrastive_loss_fn,
                contrastive_weight=extra["contrastive_weight"],
                gate_balance_weight=extra["gate_balance_weight"],
                heartbeat_every=0,
            )
            if self.args.model != "hybrid":
                # Το run_epoch του hybrid script δεν δέχεται
                # attn_entropy_weight (είναι ειδικό στο cross-attention
                # μοντέλο) -- περνιέται μόνο όταν κάνουμε search πάνω
                # στον Transformer, ώστε να είναι consistent με το
                # production default (0.05) του training script.
                run_epoch_kwargs["attn_entropy_weight"] = extra["attn_entropy_weight"]

            self.module.run_epoch(
                model, train_loader, criterion, optimizer, self.device,
                train=True, grad_clip=extra["grad_clip"],
                augment=not self.args.no_augment,
                subject_to_idx=self.subject_to_idx, grl_lambda=grl_lambda,
                **run_epoch_kwargs,
            )

            val_metrics = self.module.run_epoch(
                model, validation_loader, criterion, optimizer, self.device,
                train=False, heartbeat_every=0,
            )

            gen_score = compute_generalization_score(
                val_metrics, self.args.w_mean, self.args.w_worst, self.args.w_var,
            )
            objective_score = (
                float(val_metrics["trial_f1"])
                if self.args.objective == "trial_f1"
                else gen_score
            )

            # Ίδιο rationale με το val-smoothing στα κανονικά training
            # scripts: με μόνο ~4 validation subjects το raw GEN_SCORE
            # ανά epoch είναι θορυβώδες, οδηγώντας σε "τυχερές" επιλογές
            # trial/epoch (και σε hyperparameter importance πλήμμυρα
            # από θόρυβο, π.χ. grad_clip να φαίνεται σημαντικότερο απ'
            # ό,τι πραγματικά είναι). Χρησιμοποιούμε κυλιόμενο μέσο όρο
            # (window=3) για best-tracking/early-stop/pruning report.
            gen_score_history.append(objective_score)
            smoothed_gen_score = float(np.mean(gen_score_history[-3:]))

            print(
                f"  [Trial {trial.number}] epoch {epoch:>2}/{self.args.epochs_per_trial} | "
                f"val_trial_f1={val_metrics['trial_f1']:.4f} | "
                f"mean_subj_f1={val_metrics['subject_f1_mean']:.4f} | "
                f"worst_subj_f1={val_metrics['worst_subject_f1']:.4f} | "
                f"subj_std={val_metrics['subject_f1_std']:.4f} | "
                f"GEN_SCORE={gen_score:.4f} | objective={objective_score:.4f} | "
                f"smooth_objective={smoothed_gen_score:.4f}",
                flush=True,
            )

            trial.report(smoothed_gen_score, step=epoch)

            if smoothed_gen_score > best_trial_score:
                best_trial_score = smoothed_gen_score
                best_trial_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                best_trial_val_metrics = val_metrics
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.args.trial_patience:
                    print(f"  [Trial {trial.number}] early stop (no improvement "
                          f"{self.args.trial_patience} epochs).", flush=True)
                    break

            if trial.should_prune():
                del model, optimizer
                gc.collect()
                print(f"  [Trial {trial.number}] PRUNED at epoch {epoch}.", flush=True)
                raise optuna.exceptions.TrialPruned()

        del model, optimizer
        gc.collect()

        trial.set_user_attr("val_trial_f1", best_trial_val_metrics["trial_f1"])
        trial.set_user_attr("val_subject_f1_mean", best_trial_val_metrics["subject_f1_mean"])
        trial.set_user_attr("val_worst_subject_f1", best_trial_val_metrics["worst_subject_f1"])
        trial.set_user_attr("val_subject_f1_std", best_trial_val_metrics["subject_f1_std"])

        if best_trial_score > self.best_score:

            self.best_score = best_trial_score
            self.best_params = dict(trial.params)
            self.best_val_metrics = best_trial_val_metrics
            self.best_trial_number = trial.number

            checkpoint_path = self.output_dir / f"optuna_best_{self.args.model}.pt"
            self.output_dir.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model_state_dict": best_trial_state,
                    "config": config,
                    "params": self.best_params,
                    "generalization_score": best_trial_score,
                    "val_trial_f1": best_trial_val_metrics["trial_f1"],
                    "val_subject_f1_mean": best_trial_val_metrics["subject_f1_mean"],
                    "val_worst_subject_f1": best_trial_val_metrics["worst_subject_f1"],
                    "val_subject_f1_std": best_trial_val_metrics["subject_f1_std"],
                    "trial_number": trial.number,
                },
                checkpoint_path,
            )
            print(
                f"  >>> ΝΕΟ ΚΑΛΥΤΕΡΟ overall (Trial {trial.number}): "
                f"GEN_SCORE={best_trial_score:.4f} -- checkpoint saved to {checkpoint_path}",
                flush=True,
            )

        del best_trial_state
        gc.collect()

        return best_trial_score


def build_cli_command_hint(model_name: str, params: dict, epochs: int = 30) -> str:
    """
    Μετατρέπει τα καλύτερα hyperparameters σε ΈΤΟΙΜΗ εντολή για το
    κανονικό training script (πλήρες dataset, πολλά epochs), ώστε ο
    χρήστης να μπορεί να κάνει την τελική επιβεβαίωση χωρίς να χρειαστεί
    να αντιγράψει τιμές ένα-ένα με το χέρι.
    """

    script = "train_hybrid_baseline.py" if model_name == "hybrid" else "train_cross_attention_transformer.py"

    flags = [
        f"--epochs {epochs}",
        f"--lr {params['lr']:.6g}",
        f"--weight-decay {params['weight_decay']:.6g}",
        f"--dropout {params['dropout']:.4f}",
        f"--batch-size {params['batch_size']}",
        f"--label-smoothing {params['label_smoothing']:.4f}",
        f"--grad-clip {params['grad_clip']:.4f}",
        f"--adversarial-weight {params['adversarial_weight']:.4f}",
        f"--adversarial-lambda-max {params['adversarial_lambda_max']:.4f}",
        f"--contrastive-weight {params['contrastive_weight']:.4f}",
        f"--contrastive-temperature {params['contrastive_temperature']:.4f}",
    ]

    if model_name == "hybrid":
        flags.append(f"--embedding-dim {params['embedding_dim']}")
    else:
        ff_dim = params["ff_mult"] * params["d_model"]
        flags.append(f"--d-model {params['d_model']}")
        flags.append(f"--num-heads {params['num_heads']}")
        flags.append(f"--self-attn-layers {params['self_attn_layers']}")
        flags.append(f"--cross-attn-layers {params['cross_attn_layers']}")
        flags.append(f"--patch-size {params['patch_size']}")
        flags.append(f"--dim-feedforward {ff_dim}")
        flags.append(f"--attn-entropy-weight {params['attn_entropy_weight']:.4f}")

    return f"python {script} " + " ".join(flags)


def main():

    args = parse_args()

    module = hybrid_module if args.model == "hybrid" else transformer_module

    module.apply_quick_subset(args.num_subjects, args.trials_per_subject)

    config = Config()

    # ΚΡΙΣΙΜΟ: το build_model() διαβάζει config.MODEL_NAME για να
    # αποφασίσει ποια αρχιτεκτονική να χτίσει -- χωρίς αυτό, θα έχτιζε
    # πάντα το default ("hybrid") ανεξαρτήτως του --model flag.
    config.MODEL_NAME = args.model

    # Ίδιες "regularized" βασικές ρυθμίσεις με τα κανονικά training
    # scripts (structural defaults που ΔΕΝ αλλάζουν ανά trial -- π.χ.
    # EEG handcrafted features ON). Τα ίδια τα tunable hyperparameters
    # (lr, dropout, embedding_dim/d_model, κ.λπ.) αντικαθίστανται ανά
    # trial μέσα στο Objective.
    if args.model == "hybrid":
        config.EARLY_STOPPING_PATIENCE = 10
    else:
        config.EARLY_STOPPING_PATIENCE = 15

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if (config.DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    config.PIN_MEMORY = (device.type == "cuda")

    print("=" * 80, flush=True)
    print(f"OPTUNA HYPERPARAMETER SEARCH -- {args.model}", flush=True)
    print("=" * 80, flush=True)
    print(f"Device              : {device}", flush=True)
    print(f"Trials (n)          : {args.n_trials}", flush=True)
    print(f"Dataset cache       : {args.cache_path or 'none'}", flush=True)
    print(f"Timeout             : {args.timeout or 'None'}", flush=True)
    print(f"Epochs/Trial (max)  : {args.epochs_per_trial}", flush=True)
    print(f"Trial Patience      : {args.trial_patience}", flush=True)
    print(f"Objective           : {args.objective}", flush=True)
    print(f"Generalization Score: G = {args.w_mean}*mean_subj_f1 + "
          f"{args.w_worst}*worst_subj_f1 - {args.w_var}*subj_f1_std", flush=True)
    print(f"Num Subjects Used   : {args.num_subjects or 'ALL (32)'}", flush=True)
    print(f"Trials/Subject      : {args.trials_per_subject or 'ALL (40)'}", flush=True)
    print("=" * 80, flush=True)

    print("\nRunning data pipeline ΜΙΑ ΦΟΡΑ (θα ξαναχρησιμοποιηθεί σε ΟΛΑ τα trials)...\n", flush=True)

    t0 = time.time()

    pipeline = DataPipeline(config)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader,
        train_labels,
        train_subjects,
    ) = pipeline.run(cache_path=args.cache_path)

    print(f"\nData pipeline finished in {time.time() - t0:.1f}s\n", flush=True)

    subject_to_idx = module.build_subject_to_idx(train_dataset)
    num_subjects = len(subject_to_idx)

    print(f"Adversarial Subject Classifier: {num_subjects} train subjects\n", flush=True)

    output_dir = config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    study_name = args.study_name or f"{args.model}_cross_subject_generalization"
    storage = args.storage or f"sqlite:///{(output_dir / 'optuna_studies.db').as_posix()}"

    print(f"Optuna study name   : {study_name}", flush=True)
    print(f"Optuna storage      : {storage}", flush=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=2, interval_steps=1,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    objective = Objective(
        args, config, module, device,
        train_dataset, validation_dataset, test_dataset,
        train_labels, subject_to_idx, num_subjects, output_dir,
    )

    search_t0 = time.time()

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    search_minutes = (time.time() - search_t0) / 60.0

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print("\n" + "=" * 80, flush=True)
    print("OPTUNA SEARCH FINISHED", flush=True)
    print("=" * 80, flush=True)
    print(f"Total trials        : {len(study.trials)} "
          f"({len(completed)} completed, {len(pruned)} pruned)", flush=True)
    print(f"Search duration     : {search_minutes:.1f} min", flush=True)
    print(f"Best objective value  : {study.best_value:.4f}", flush=True)
    print(f"Best trial number   : {study.best_trial.number}", flush=True)
    print("Best hyperparameters:", flush=True)
    for k, v in study.best_params.items():
        print(f"    {k:>24}: {v}", flush=True)
    print("Best trial val metrics:", flush=True)
    for k, v in study.best_trial.user_attrs.items():
        print(f"    {k:>24}: {v:.4f}", flush=True)

    # ΣΗΜΑΝΤΙΚΟ: το ΚΑΛΥΤΕΡΟ trial πάνω σε ένα ΜΙΚΡΟ subset/validation
    # split μπορεί απλά να είναι "τυχερό" πάνω στα λίγα validation
    # subjects εκείνου του subset -- όχι μια πραγματικά ανώτερη περιοχή
    # υπερπαραμέτρων. Γι' αυτό τυπώνουμε ΚΑΙ τα Top-5 trials: αν οι
    # τιμές των hyperparameters διαφέρουν πολύ μεταξύ τους, αυτό είναι
    # ένδειξη υψηλού noise στο search -- προτιμήστε τιμές που
    # επαναλαμβάνονται σε πολλά από τα καλύτερα trials, αντί να
    # εμπιστευτείτε τυφλά μόνο το Νο1.
    top_n = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    print("\nTop-5 trials (έλεγξε αν οι τιμές συγκλίνουν -- αν διαφέρουν "
          "πολύ, το search έχει πολύ noise):", flush=True)
    for rank, t in enumerate(top_n, start=1):
        print(f"  #{rank} trial {t.number} | score={t.value:.4f} | {t.params}", flush=True)
    print("=" * 80, flush=True)

    best_info = {
        "model": args.model,
        "study_name": study_name,
        "storage": storage,
        "best_objective_value": study.best_value,
        "objective": args.objective,
        "best_trial_number": study.best_trial.number,
        "best_params": study.best_params,
        "best_trial_val_metrics": study.best_trial.user_attrs,
        "top_5_trials": [
            {"trial_number": t.number, "score": t.value, "params": t.params}
            for t in top_n
        ],
        "n_trials_requested": args.n_trials,
        "n_trials_completed": len(completed),
        "n_trials_pruned": len(pruned),
        "search_minutes": search_minutes,
        "weights": {"w_mean": args.w_mean, "w_worst": args.w_worst, "w_var": args.w_var},
    }

    json_path = output_dir / f"optuna_best_params_{args.model}.json"
    json_path.write_text(json.dumps(best_info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nBest params saved to: {json_path}", flush=True)

    try:
        import optuna.visualization as vis

        vis.plot_optimization_history(study).write_html(
            str(output_dir / f"optuna_history_{args.model}.html")
        )
        vis.plot_param_importances(study).write_html(
            str(output_dir / f"optuna_importance_{args.model}.html")
        )
        print(f"Optuna plots saved to: {output_dir}\\optuna_history_{args.model}.html "
              f"και optuna_importance_{args.model}.html", flush=True)
    except Exception as exc:
        print(f"(Plot generation skipped: {exc})", flush=True)

    print(
        f"\nΤο καλύτερο μοντέλο (checkpoint) αποθηκεύτηκε ήδη κατά το "
        f"search στο: {output_dir / f'optuna_best_{args.model}.pt'}",
        flush=True,
    )

    print("\nΓια ΤΕΛΙΚΗ επιβεβαίωση στο πλήρες dataset (30 epochs, ίδιο "
          "evaluation protocol με τα προηγούμενα runs), τρέξε:\n", flush=True)
    print(build_cli_command_hint(args.model, study.best_params), flush=True)


if __name__ == "__main__":
    main()
