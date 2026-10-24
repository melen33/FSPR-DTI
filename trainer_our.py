import sys
import torch
import torch.nn as nn
from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    confusion_matrix, precision_recall_curve,
    accuracy_score, f1_score, precision_score,
    matthews_corrcoef
)
from scipy.stats import pearsonr, spearmanr
from prettytable import PrettyTable
from tqdm import tqdm
# from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast
from utils.relation_alignment import compute_relation_alignment_loss
from conformation_pretraining.joint_response import structural_response_loss_from_batch

@dataclass(frozen=True)
class RegressionTargetScaler:
    """Affine target scaler fitted strictly on the temporal training partition."""

    mean: float
    std: float

    @classmethod
    def fit(cls, values):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
            raise ValueError("Regression training targets must be a non-empty finite vector.")
        std = float(array.std(ddof=0))
        return cls(mean=float(array.mean()), std=std if std > 1e-12 else 1.0)

    def transform(self, values):
        return (values - self.mean) / self.std

    def inverse(self, values):
        return values * self.std + self.mean

    def as_dict(self):
        return {"mean": self.mean, "std": self.std, "fit_partition": "train"}


def regression_metrics(y_true, y_pred):
    """Compute original-scale temporal affinity metrics with explicit finiteness checks."""
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.ndim != 1 or prediction.ndim != 1 or len(truth) != len(prediction):
        raise ValueError("Regression metric inputs must be equally sized vectors.")
    if len(truth) < 2 or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Regression metrics require at least two finite predictions and labels.")
    return {
        "pcc": float(pearsonr(truth, prediction)[0]),
        "spearman": float(spearmanr(truth, prediction)[0]),
        "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
        "mae": float(np.mean(np.abs(truth - prediction))),
    }


def binary_validation_selection_score(auroc, auprc, selection_metric):
    """Return the validation-only checkpoint-selection score for binary DTI."""
    selection_metric = str(selection_metric)
    if selection_metric == "auroc":
        return float(auroc)
    if selection_metric == "auroc_plus_auprc":
        return float(auroc) + float(auprc)
    raise ValueError(
        "Unknown binary validation selection metric: %s" % selection_metric
    )




class Trainer(object):
    CHECKPOINT_FORMAT_VERSION = 1

    def __init__(self, model, optim, device,
                 train_dataloader, val_dataloader, test_dataloader,
                 output_path, config, structural_dataloader=None,
                 structural_weight=0.0, structural_geometry_weight=1.0, structural_interval=1,
                 structural_margin=0.1, structural_ranking_weight=1.0,
                 structural_site_ranking_weight=0.0, structural_site_ranking_min_gap=0.1,
                 structural_residue_ranking_weight=0.0, structural_residue_ranking_min_gap=0.10, structural_residue_ranking_temperature=0.20,
                 structural_mode="scalar_rmsd", structural_label_perturbation="none",
                 structural_gearnet_weight=0.0, structural_gearnet_semantic_weight=0.0,
                 structural_gearnet_global_weight=0.0,
                 structural_gearnet_relational_weight=0.0, structural_gearnet_relational_temperature=0.1,
                 structural_gearnet_direct_weight=0.0, structural_gearnet_pair_weight=0.0, structural_gearnet_pair_temperature=0.5,
                 structural_loss_normalization="none", structural_loss_ema_decay=0.99,
                 structural_contact_weight_mode="none", structural_contact_weight_alpha=1.0,
                 structural_contact_weight_sigma=4.5, structural_contact_weight_shuffle_seed=0,
                 validation_only=False, zero_epoch_eval=False, diagnostic_eval_each_epoch=False,
                 prediction_export_path=None):

        self.model = model
        self.optim = optim
        self.device = device
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.epochs = config.TRAIN.MAX_EPOCH
        self.config = config
        self.task_mode = str(getattr(config, "task_mode", "binary"))
        if self.task_mode not in {"binary", "regression"}:
            raise ValueError("task_mode must be binary or regression")
        self.validation_selection_metric = str(
            getattr(config.TRAIN, "VALIDATION_SELECTION_METRIC", "auroc")
        )
        if self.task_mode == "binary" and self.validation_selection_metric not in {
            "auroc", "auroc_plus_auprc",
        }:
            raise ValueError(
                "TRAIN.VALIDATION_SELECTION_METRIC must be 'auroc' or "
                "'auroc_plus_auprc' for binary training."
            )
        self.train_dataloader = train_dataloader
        if self.task_mode == "regression":
            train_ids = list(getattr(train_dataloader.dataset, "list_IDs", train_dataloader.dataset.df.index))
            train_targets = train_dataloader.dataset.df.iloc[train_ids]["Y"].to_numpy(dtype=np.float64)
            self.target_scaler = RegressionTargetScaler.fit(train_targets)
            self.criterion = nn.SmoothL1Loss()
        else:
            self.target_scaler = None
            self.criterion = nn.CrossEntropyLoss()
        self.lambda_mlm = float(getattr(config.TRAIN, "lambda_mlm", 1.0))
        self.use_amp = bool(
            getattr(config.TRAIN, "USE_AMP", True)
            and device.type == "cuda"
        )
        self.grad_clip = float(getattr(config.TRAIN, "GRAD_CLIP", 5.0))
        self.compute_if_score = bool(getattr(config, "compute_if_score", False))
        self.prediction_export_path = prediction_export_path
        self.relation_alignment_enabled = bool(getattr(config, "relation_alignment_enabled", False))
        self.alignment_target = str(getattr(config, "alignment_target", "none"))
        if not self.relation_alignment_enabled:
            self.alignment_target = "none"
        if self.task_mode == "regression" and self.alignment_target != "none":
            raise ValueError(
                "Relation alignment is defined for binary DTI labels and must be disabled "
                "for temporal affinity regression."
            )
        self.relation_alignment_weight = float(getattr(config, "relation_alignment_weight", 0.05))
        self.relation_alignment_side = str(getattr(config, "relation_alignment_side", "both"))
        if self.relation_alignment_side not in {"both", "drug", "protein"}:
            raise ValueError("relation_alignment_side must be one of 'both', 'drug', or 'protein'.")
        self.teacher_temperature = float(getattr(config, "teacher_temperature", 0.1))
        self.student_temperature = float(getattr(config, "student_temperature", 0.1))
        self.positive_only_alignment = bool(getattr(config, "positive_only_alignment", True))
        self.min_positive_for_alignment = int(getattr(config, "min_positive_for_alignment", 3))
        self.if_score_alpha = float(getattr(config, "if_score_alpha", 0.5))
        self.save_if_score_path = str(getattr(config, "save_if_score_path", "") or "")
        self.target_protein_for_case_study = str(
            getattr(config, "target_protein_for_case_study", "") or ""
        )
        # Keep terminal output compact: no per-batch tqdm bars by default.
        # Set TRAIN.QUIET_PROGRESS=false in config if you ever want tqdm back.
        self.quiet_progress = bool(getattr(config.TRAIN, "QUIET_PROGRESS", True))

        self.current_epoch = 0
        self.step = 0

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.structural_dataloader = structural_dataloader
        self.structural_weight = float(structural_weight)
        self.structural_geometry_weight = float(structural_geometry_weight)
        self.structural_interval = int(structural_interval)
        self.structural_margin = float(structural_margin)
        self.structural_ranking_weight = float(structural_ranking_weight)
        self.structural_site_ranking_weight = float(structural_site_ranking_weight)
        self.structural_site_ranking_min_gap = float(structural_site_ranking_min_gap)
        self.structural_residue_ranking_weight = float(structural_residue_ranking_weight)
        self.structural_residue_ranking_min_gap = float(structural_residue_ranking_min_gap)
        self.structural_residue_ranking_temperature = float(structural_residue_ranking_temperature)
        if self.structural_residue_ranking_weight < 0.0:
            raise ValueError("structural_residue_ranking_weight must be non-negative")
        if self.structural_residue_ranking_min_gap < 0.0:
            raise ValueError("structural_residue_ranking_min_gap must be non-negative")
        if self.structural_residue_ranking_temperature <= 0.0:
            raise ValueError("structural_residue_ranking_temperature must be positive")
        if self.structural_site_ranking_weight < 0.0:
            raise ValueError("structural_site_ranking_weight must be non-negative")
        if self.structural_site_ranking_min_gap < 0.0:
            raise ValueError("structural_site_ranking_min_gap must be non-negative")
        self.structural_mode = str(structural_mode)
        if self.structural_mode not in {"scalar_rmsd", "pocket_residue", "gearnet_only", "full_protein_residue", "pocket_rms"}:
            raise ValueError("structural_mode must be scalar_rmsd, pocket_residue, or gearnet_only")
        self.structural_gearnet_weight = float(structural_gearnet_weight)
        self.structural_gearnet_semantic_weight = float(structural_gearnet_semantic_weight)
        self.structural_gearnet_global_weight = float(structural_gearnet_global_weight)
        self.structural_gearnet_relational_weight = float(structural_gearnet_relational_weight)
        self.structural_gearnet_relational_temperature = float(structural_gearnet_relational_temperature)
        self.structural_gearnet_direct_weight = float(structural_gearnet_direct_weight)
        self.structural_gearnet_pair_weight = float(structural_gearnet_pair_weight)
        self.structural_gearnet_pair_temperature = float(structural_gearnet_pair_temperature)
        self.structural_loss_normalization = str(structural_loss_normalization)
        self.structural_loss_ema_decay = float(structural_loss_ema_decay)
        self.structural_contact_weight_mode = str(structural_contact_weight_mode)
        self.structural_contact_weight_alpha = float(structural_contact_weight_alpha)
        self.structural_contact_weight_sigma = float(structural_contact_weight_sigma)
        self.structural_contact_weight_shuffle_seed = int(structural_contact_weight_shuffle_seed)
        if self.structural_contact_weight_mode not in {"none", "contact", "residue_shuffle"}:
            raise ValueError("structural_contact_weight_mode must be none, contact, or residue_shuffle")
        if self.structural_contact_weight_alpha < 0.0:
            raise ValueError("structural_contact_weight_alpha must be non-negative")
        if self.structural_contact_weight_sigma <= 0.0:
            raise ValueError("structural_contact_weight_sigma must be positive")
        if self.structural_loss_normalization not in {"none", "ema"}:
            raise ValueError("structural_loss_normalization must be none or ema")
        if not 0.0 <= self.structural_loss_ema_decay < 1.0:
            raise ValueError("structural_loss_ema_decay must be in [0, 1)")
        self._structural_loss_ema = {}
        if self.structural_gearnet_semantic_weight < 0.0:
            raise ValueError("structural_gearnet_semantic_weight must be non-negative")
        if self.structural_gearnet_global_weight < 0.0:
            raise ValueError("structural_gearnet_global_weight must be non-negative")
        if self.structural_gearnet_global_weight > 0.0 and self.structural_mode != "pocket_residue":
            raise ValueError("structural_gearnet_global_weight requires pocket_residue mode")
        if self.structural_gearnet_semantic_weight > 0.0 and self.structural_mode != "pocket_residue":
            raise ValueError("structural_gearnet_semantic_weight requires pocket_residue mode")
        if self.structural_gearnet_relational_weight < 0.0:
            raise ValueError("structural_gearnet_relational_weight must be non-negative")
        if self.structural_gearnet_relational_temperature <= 0.0:
            raise ValueError("structural_gearnet_relational_temperature must be positive")
        if self.structural_gearnet_direct_weight < 0.0:
            raise ValueError("structural_gearnet_direct_weight must be non-negative")
        if self.structural_gearnet_pair_weight < 0.0:
            raise ValueError("structural_gearnet_pair_weight must be non-negative")
        if self.structural_gearnet_pair_temperature <= 0.0:
            raise ValueError("structural_gearnet_pair_temperature must be positive")
        if (self.structural_gearnet_direct_weight > 0.0 or self.structural_gearnet_pair_weight > 0.0) and self.structural_mode != "pocket_residue":
            raise ValueError("response-bound GearNet distillation requires pocket_residue mode")
        if self.structural_gearnet_semantic_weight > 0.0 and self.structural_gearnet_relational_weight > 0.0:
            raise ValueError("direct and relational GearNet semantic weights are mutually exclusive")
        if self.structural_gearnet_relational_weight > 0.0 and self.structural_mode != "pocket_residue":
            raise ValueError("structural_gearnet_relational_weight requires pocket_residue mode")
        if self.structural_gearnet_weight < 0.0:
            raise ValueError("structural_gearnet_weight must be non-negative")
        if self.structural_gearnet_weight > 0.0 and self.structural_mode not in {"pocket_residue", "gearnet_only"}:
            raise ValueError("structural_gearnet_weight requires pocket_residue or gearnet_only mode")
        if self.structural_mode == "gearnet_only" and self.structural_gearnet_weight <= 0.0:
            raise ValueError("gearnet_only requires a positive structural_gearnet_weight")
        self.structural_label_perturbation = str(structural_label_perturbation)
        if self.structural_label_perturbation not in {"none", "batch_shuffle", "global_random_derangement"}:
            raise ValueError("structural_label_perturbation must be none, batch_shuffle, or global_random_derangement")
        self.structural_label_perturbation_seed = int(config.TRAIN.SEED)
        self.structural_target_random_deletion_ratio = float(
            getattr(config.TRAIN, "TARGET_RANDOM_DROP_RATIO", 0.0)
        )
        if self.structural_weight < 0:
            raise ValueError("structural_weight must be non-negative")
        if self.structural_geometry_weight < 0:
            raise ValueError("structural_geometry_weight must be non-negative")
        if self.structural_interval < 1:
            raise ValueError("structural_interval must be at least 1")
        self.structural_enabled = (
            self.structural_dataloader is not None and self.structural_weight > 0.0
        )
        self.validation_only = bool(validation_only)
        self.zero_epoch_eval = bool(zero_epoch_eval)
        self.diagnostic_eval_each_epoch = bool(diagnostic_eval_each_epoch)

        self.nb_training = len(self.train_dataloader)

        self.best_state_dict = None
        self.best_epoch = None
        self.best_auroc = -float("inf") if self.task_mode == "regression" else 0.0
        self.best_validation_auprc = None
        self.best_selection_score = -float("inf")

        # ===== Early Stopping =====
        self.early_stop = getattr(config.TRAIN, "EARLY_STOP", True)
        self.early_stop_patience = getattr(config.TRAIN, "EARLY_STOP_PATIENCE", 5)
        self.early_stop_delta = getattr(config.TRAIN, "EARLY_STOP_DELTA", 1e-4)

        self.no_improve_epochs = 0
        self.best_val_metric = -float("inf")

        self.train_loss_epoch = []
        self.val_loss_epoch = []
        self.val_auroc_epoch = []
        self.test_metrics = {}

        self.output_dir = output_path

        self.train_table = PrettyTable(["# Epoch", "Train_loss"])
        if self.task_mode == "regression":
            self.val_table = PrettyTable(["# Epoch", "PCC", "Spearman"])
            self.test_table = PrettyTable([
                "# Best Epoch", "PCC", "Spearman", "RMSE", "MAE"
            ])
        else:
            self.val_table = PrettyTable(["# Epoch", "AUROC", "AUPRC"])
            self.test_table = PrettyTable([
                "# Best Epoch", "AUROC", "AUPRC",
                "Precision", "Recall", "F1",
                "Specificity", "Accuracy", "MCC", "Threshold"
            ])

    # =========================
    # 核心：保存完整模型状态
    # =========================
    @staticmethod
    def get_trainable_state_dict(model):
        return {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    def save_best_checkpoint(self):
        if self.best_state_dict is None or self.best_epoch is None:
            raise RuntimeError("Cannot save checkpoint before a best model is selected.")

        plain_config = OmegaConf.to_container(self.config, resolve=True)
        model_config = {
            key: value
            for key, value in plain_config.items()
            if key not in {"TRAIN", "TEST"}
        }
        payload = {
            "optimization_provenance": {
                "configured_batch_size": int(plain_config["TRAIN"]["BATCH_SIZE"]),
                "configured_gradient_accumulation_steps": int(
                    plain_config["TRAIN"].get("GRADIENT_ACCUMULATION_STEPS", 1)
                ),
                "gradient_accumulation_implemented": False,
                "optimizer_step_interval_batches": 1,
                "actual_effective_batch_size": int(plain_config["TRAIN"]["BATCH_SIZE"]),
            },
            "format_version": self.CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": self.best_state_dict,
            "config": plain_config,
            "training_config": plain_config["TRAIN"],
            "task_mode": getattr(self, "task_mode", "binary"),
            "target_scaler": (
                self.target_scaler.as_dict()
                if getattr(self, "target_scaler", None) is not None else None
            ),
            "best_validation_metric": float(self.best_auroc),
            "model_config": model_config,
            "best_epoch": int(self.best_epoch),
            "best_validation_auroc": float(self.best_auroc),
            "best_validation_auprc": (
                None if getattr(self, "best_validation_auprc", None) is None
                else float(self.best_validation_auprc)
            ),
            "validation_selection_metric": getattr(
                self, "validation_selection_metric", "auroc"
            ),
            "best_validation_selection_score": float(
                getattr(self, "best_selection_score", self.best_auroc)
            ),
        }

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "best_checkpoint.pt"
        temporary_path = output_dir / ".best_checkpoint.pt.tmp"
        torch.save(payload, temporary_path)
        temporary_path.replace(checkpoint_path)
        return checkpoint_path

    def evaluate_checkpoint(self, checkpoint_path):
        """Restore a saved best checkpoint and evaluate target test only.

        This intentionally does not invoke ``train_epoch`` or write a checkpoint.
        It is intended for post-selection diagnostic reporting only.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint was not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
            raise ValueError("Checkpoint must contain a model_state_dict mapping.")
        checkpoint_task_mode = str(payload.get("task_mode", "binary"))
        if checkpoint_task_mode != self.task_mode:
            raise ValueError(
                f"Checkpoint task_mode={checkpoint_task_mode!r} does not match "
                f"current task_mode={self.task_mode!r}."
            )
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()
        best_epoch = int(payload.get("best_epoch", -1))
        if best_epoch < 0:
            raise ValueError("Checkpoint best_epoch must be a non-negative integer.")

        if self.task_mode == "regression":
            scaler = payload.get("target_scaler")
            if not isinstance(scaler, dict) or "mean" not in scaler or "std" not in scaler:
                raise ValueError("Regression checkpoint must contain target_scaler mean and std.")
            self.target_scaler = RegressionTargetScaler(
                mean=float(scaler["mean"]), std=float(scaler["std"])
            )
            metrics = dict(self.test(dataloader="test"))
            metrics["best_epoch"] = best_epoch
            metrics["best_validation_pcc"] = float(
                payload.get("best_validation_metric", payload.get("best_validation_auroc", float("nan")))
            )
            print(
                f"Checkpoint-only target test | Best Epoch {best_epoch} | "
                f"PCC {metrics['pcc']:.4f} | Spearman {metrics['spearman']:.4f} | "
                f"RMSE {metrics['rmse']:.4f} | MAE {metrics['mae']:.4f}",
                flush=True,
            )
            return metrics, best_epoch

        (auroc, auprc, f1, precision, recall,
         specificity, accuracy, mcc, threshold) = self.test(dataloader="test")
        metrics = {
            "auroc": auroc,
            "auprc": auprc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "accuracy": accuracy,
            "mcc": mcc,
            "threshold": threshold,
            "best_epoch": best_epoch,
            "validation_auroc": float(
                payload.get("best_validation_auroc", payload.get("best_validation_metric", float("nan")))
            ),
        }
        print(
            f"Checkpoint-only target test | Best Epoch {best_epoch} | "
            f"AUROC {auroc:.4f} | AUPRC {auprc:.4f} | F1 {f1:.4f} | "
            f"ACC {accuracy:.4f} | MCC {mcc:.4f} | Precision {precision:.4f} | "
            f"Recall {recall:.4f} | Specificity {specificity:.4f} | Threshold {threshold:.4f}",
            flush=True,
        )
        return metrics, best_epoch

    def train(self):
        float2str = lambda x: '%0.4f' % x

        if getattr(self, "zero_epoch_eval", False):
            self.model.to(self.device)
            self.model.eval()
            if self.task_mode == "regression":
                validation_result = self.test(dataloader="val")
                test_result = self.test(dataloader="test")
                self.test_metrics = dict(test_result)
                self.test_metrics.update({"validation_pcc": validation_result["pcc"], "validation_spearman": validation_result["spearman"], "best_epoch": 0})
                print(f"Zero-epoch evaluation | Val PCC {validation_result['pcc']:.4f} | Test PCC {test_result['pcc']:.4f} | Spearman {test_result['spearman']:.4f}", flush=True)
                return self.test_metrics, 0
            val_auroc, val_auprc = self.test(dataloader="val")
            (auroc, auprc, f1, precision, recall,
             specificity, accuracy, mcc, thred_optim) = self.test(dataloader="test")
            self.test_metrics = {"auroc": auroc, "auprc": auprc, "precision": precision, "recall": recall, "f1": f1, "specificity": specificity, "accuracy": accuracy, "mcc": mcc, "threshold": thred_optim, "validation_auroc": val_auroc, "validation_auprc": val_auprc, "best_epoch": 0}
            print(f"Zero-epoch evaluation | Val AUROC {val_auroc:.4f} | Test AUROC {auroc:.4f} | AUPRC {auprc:.4f}", flush=True)
            return self.test_metrics, 0
        for _ in range(self.epochs):
            self.current_epoch += 1
            print(f"Epoch {self.current_epoch:03d}/{self.epochs:03d} | training...", flush=True)
            train_loss = self.train_epoch()

            self.train_loss_epoch.append(train_loss)
            self.train_table.add_row(
                [f"epoch {self.current_epoch}", float2str(train_loss)]
            )

            validation_result = self.test(dataloader="val")
            if self.task_mode == "regression":
                auroc = validation_result["pcc"]
                auprc = validation_result["spearman"]
                selection_score = auroc
            else:
                auroc, auprc = validation_result
                selection_score = binary_validation_selection_score(
                    auroc, auprc, getattr(self, "validation_selection_metric", "auroc")
                )
            self.val_auroc_epoch.append(auroc)

            self.val_table.add_row(
                [f"epoch {self.current_epoch}",
                 float2str(auroc), float2str(auprc)]
            )
            if getattr(self, "diagnostic_eval_each_epoch", False):
                if self.task_mode == "regression":
                    test_result = self.test(dataloader="test")
                    diagnostic_row = {"epoch": self.current_epoch, "train_loss": train_loss, "validation_pcc": auroc, "validation_spearman": auprc, "test_pcc": test_result["pcc"], "test_spearman": test_result["spearman"], "test_rmse": test_result["rmse"], "test_mae": test_result["mae"]}
                else:
                    (test_auroc, test_auprc, test_f1, test_precision, test_recall,
                     test_specificity, test_accuracy, test_mcc, test_threshold) = self.test(dataloader="test")
                    diagnostic_row = {"epoch": self.current_epoch, "train_loss": train_loss, "validation_auroc": auroc, "validation_auprc": auprc, "test_auroc": test_auroc, "test_auprc": test_auprc, "test_precision": test_precision, "test_recall": test_recall, "test_f1": test_f1, "test_specificity": test_specificity, "test_accuracy": test_accuracy, "test_mcc": test_mcc, "test_threshold": test_threshold}
                trace_path = os.path.join(self.output_dir, "epoch_diagnostic_metrics.csv")
                pd.DataFrame([diagnostic_row]).to_csv(trace_path, mode="a", header=not os.path.exists(trace_path), index=False)
            previous_best_selection_score = getattr(
                self, "best_val_metric", -float("inf")
            )
            # ===== best model =====
            if selection_score > previous_best_selection_score:
                self.best_auroc = auroc
                self.best_validation_auprc = auprc
                self.best_selection_score = selection_score
                self.best_epoch = self.current_epoch
                self.best_state_dict = self.get_trainable_state_dict(self.model)
                self.best_val_metric = selection_score

            # ===== Early Stopping =====
            if self.early_stop:
                if selection_score > previous_best_selection_score + self.early_stop_delta:
                    self.no_improve_epochs = 0
                else:
                    self.no_improve_epochs += 1

                if self.no_improve_epochs >= self.early_stop_patience:
                    print(
                        f"\n Early Stopping Triggered! "
                        f"No improvement for {self.early_stop_patience} epochs.\n"
                    )
                    break

            primary_metric = "PCC" if self.task_mode == "regression" else "AUROC"
            secondary_metric = "Spearman" if self.task_mode == "regression" else "AUPRC"
            selection_text = (
                "PCC" if self.task_mode == "regression"
                else getattr(self, "validation_selection_metric", "auroc")
            )
            print(
                f"Epoch {self.current_epoch:03d}/{self.epochs:03d} | "
                f"train_loss {train_loss:.4f} | "
                f"val_{primary_metric} {auroc:.4f} | val_{secondary_metric} {auprc:.4f} | "
                f"selection[{selection_text}] {selection_score:.4f}",
                flush=True,
            )

        if self.validation_only:
            self.model.load_state_dict(self.best_state_dict, strict=False)
            self.model.to(self.device)
            if self.task_mode == "regression":
                result = {"best_validation_pcc": float(self.best_auroc), "best_epoch": int(self.best_epoch)}
                print(f"Validation-only complete | Best Epoch {self.best_epoch} | Val PCC {self.best_auroc:.4f} | test evaluation skipped", flush=True)
            else:
                result = {"best_validation_auroc": float(self.best_auroc), "best_epoch": int(self.best_epoch)}
                print(f"Validation-only complete | Best Epoch {self.best_epoch} | Val AUROC {self.best_auroc:.4f} | test evaluation skipped", flush=True)
            self.save_validation_result(result)
            return result, self.best_epoch
        # =========================
        # Test with best weights
        # =========================
        self.model.load_state_dict(self.best_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        if self.task_mode == "regression":
            regression_result = self.test(dataloader="test")
            self.test_metrics = dict(regression_result)
            self.test_metrics["best_epoch"] = self.best_epoch
            self.test_metrics["best_validation_pcc"] = float(self.best_auroc)
            self.test_table.add_row(list(map(float2str, [
                self.best_epoch, regression_result["pcc"], regression_result["spearman"],
                regression_result["rmse"], regression_result["mae"]
            ])))
            print(
                f"Test at Best Epoch {self.best_epoch} | "
                f"PCC {regression_result['pcc']:.4f} | "
                f"Spearman {regression_result['spearman']:.4f} | "
                f"RMSE {regression_result['rmse']:.4f} | MAE {regression_result['mae']:.4f}",
                flush=True,
            )
            self.save_result()
            return self.test_metrics, self.best_epoch


        (auroc, auprc, f1, precision, recall,
         specificity, accuracy, mcc, thred_optim) = self.test(dataloader="test")

        self.test_table.add_row(list(map(float2str, [
            self.best_epoch, auroc, auprc,
            precision, recall, f1,
            specificity, accuracy, mcc,
            thred_optim
        ])))

        print(
            f"Test at Best Epoch {self.best_epoch} | "
            f"AUROC {auroc:.4f} | AUPRC {auprc:.4f} | "
            f"F1 {f1:.4f} | ACC {accuracy:.4f} | MCC {mcc:.4f} | "
            f"Precision {precision:.4f} | Recall {recall:.4f} | "
            f"Specificity {specificity:.4f} | Threshold {thred_optim:.4f}",
            flush=True,
        )

        self.test_metrics = {
            "auroc": auroc,
            "auprc": auprc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "accuracy": accuracy,
            "mcc": mcc,
            "threshold": thred_optim,
            "best_epoch": self.best_epoch,
            "best_validation_auroc": float(self.best_auroc),
            "best_validation_auprc": float(getattr(self, "best_validation_auprc", auprc)),
            "validation_selection_metric": getattr(self, "validation_selection_metric", "auroc"),
            "best_validation_selection_score": float(
                getattr(self, "best_selection_score", self.best_auroc)
            ),
        }

        self.save_result()
        return self.test_metrics, self.best_epoch

    def save_validation_result(self, result):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.config.TRAIN.SAVE_MODEL:
            self.save_best_checkpoint()
        metric_key = "best_validation_pcc" if self.task_mode == "regression" else "best_validation_auroc"
        torch.save({
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_metric": self.val_auroc_epoch,
            metric_key: float(result[metric_key]),
            "best_epoch": int(result["best_epoch"]),
            "task_mode": self.task_mode,
            "target_scaler": self.target_scaler.as_dict() if self.target_scaler is not None else None,
            "validation_only": True,
            "config": self.config,
        }, os.path.join(self.output_dir, "result_metrics.pt"))
    def save_result(self):
        os.makedirs(self.output_dir, exist_ok=True)

        #  只保存 LoRA / 可训练参数
        if self.config.TRAIN.SAVE_MODEL:
            self.save_best_checkpoint()

        # metrics
        torch.save({
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_metric": self.val_auroc_epoch,
            "task_mode": self.task_mode,
            "target_scaler": self.target_scaler.as_dict() if self.target_scaler is not None else None,
            "test_metrics": self.test_metrics,
            "config": self.config
        }, os.path.join(self.output_dir, "result_metrics.pt"))

        # tables
        with open(os.path.join(self.output_dir, "train_markdowntable.txt"), "w") as f:
            f.write(self.train_table.get_string())

        with open(os.path.join(self.output_dir, "valid_markdowntable.txt"), "w") as f:
            f.write(self.val_table.get_string())

        with open(os.path.join(self.output_dir, "test_markdowntable.txt"), "w") as f:
            f.write(self.test_table.get_string())

    @staticmethod
    def _loss_item(value):
        if value is None:
            return 0.0
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    @staticmethod
    def _usage_summary(usage_stats):
        if not usage_stats:
            return "off"
        parts = []
        for level, stat in enumerate(usage_stats):
            parts.append(f"q{level + 1}:{stat['used_codes']}/{stat['perplexity']:.1f}")
        return " ".join(parts)

    def _relation_alignment_output(self, output, labels):
        if self.alignment_target == "none":
            zero = output['logits'].sum() * 0.0
            return {
                'relation_loss': zero,
                'drug_relation_loss': zero.detach() + zero,
                'protein_relation_loss': zero.detach() + zero,
                'positive_count': torch.as_tensor(0, device=zero.device),
                'valid_relation_batch': False,
                'drug_response_offdiag_cos': zero.detach(),
                'protein_response_offdiag_cos': zero.detach(),
                'response_mean_std': zero.detach(),
            }
        required = [
            'drug_response', 'protein_response',
            'drug_teacher_embedding', 'protein_teacher_embedding',
        ]
        missing = [key for key in required if key not in output]
        if missing:
            raise KeyError(f"Relation alignment enabled but model output is missing: {missing}")
        return compute_relation_alignment_loss(
            drug_response=output['drug_response'],
            protein_response=output['protein_response'],
            drug_teacher=output['drug_teacher_embedding'],
            protein_teacher=output['protein_teacher_embedding'],
            labels=labels,
            teacher_temperature=self.teacher_temperature,
            student_temperature=self.student_temperature,
            positive_only=self.positive_only_alignment,
            min_positive=self.min_positive_for_alignment,
            alignment_side=self.relation_alignment_side,
        )

    def _add_auxiliary_losses(self, output, task_loss, labels=None):
        relation_out = None
        relation_loss = task_loss.new_tensor(0.0)
        if self.alignment_target != "none":
            if labels is None:
                raise ValueError("labels are required for relation alignment.")
            relation_out = self._relation_alignment_output(output, labels)
            relation_loss = relation_out['relation_loss']
        total = task_loss + self.relation_alignment_weight * relation_loss
        if relation_out is not None:
            output.update({
                'relation_loss': relation_out['relation_loss'].detach(),
                'drug_relation_loss': relation_out['drug_relation_loss'].detach(),
                'protein_relation_loss': relation_out['protein_relation_loss'].detach(),
                'relation_positive_count': relation_out['positive_count'].detach(),
                'valid_relation_batch': relation_out['valid_relation_batch'],
                'drug_response_offdiag_cos': relation_out['drug_response_offdiag_cos'].detach(),
                'protein_response_offdiag_cos': relation_out['protein_response_offdiag_cos'].detach(),
                'response_mean_std': relation_out['response_mean_std'].detach(),
            })
        else:
            zero = task_loss.new_tensor(0.0)
            output.update({
                'relation_loss': zero.detach(),
                'drug_relation_loss': zero.detach(),
                'protein_relation_loss': zero.detach(),
                'relation_positive_count': zero.detach(),
                'valid_relation_batch': False,
                'drug_response_offdiag_cos': zero.detach(),
                'protein_response_offdiag_cos': zero.detach(),
                'response_mean_std': zero.detach(),
            })
        return total

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0.0
        relation_loss_epoch = 0.0
        drug_relation_loss_epoch = 0.0
        protein_relation_loss_epoch = 0.0
        positive_count_epoch = 0.0
        valid_relation_batches = 0
        drug_cos_epoch = 0.0
        protein_cos_epoch = 0.0
        response_std_epoch = 0.0
        # scaler = GradScaler()
        scaler = GradScaler(device='cuda', enabled=self.use_amp)
        loop = tqdm(
            self.train_dataloader,
            file=sys.stdout,
            colour="#bdbdbd",
            disable=self.quiet_progress,
            leave=False,
        )
        if not self.quiet_progress:
            loop.set_description(f"Train Epoch [{self.current_epoch}/{self.epochs}]")
        if self.structural_enabled:
            structural_sampler = getattr(self.structural_dataloader, "batch_sampler", None)
            if hasattr(structural_sampler, "set_epoch"):
                structural_sampler.set_epoch(self.current_epoch)
        structural_iterator = iter(self.structural_dataloader) if self.structural_enabled else None
        structural_loss_epoch = 0.0
        structural_semantic_epoch = 0.0
        structural_steps = 0
        structural_residue_rank_epoch = 0.0
        structural_residue_rank_pairs = 0
        structural_residue_rank_skipped = 0

        for step, batch in enumerate(loop):
            self.optim.zero_grad()
            self.step += 1

            # with autocast():
            with autocast(device_type='cuda', enabled=self.use_amp):
                input_drugs = batch['batch_inputs_drug'].to(self.device)
                input_proteins = batch['batch_inputs_pr']['input_ids'].to(self.device)
                pr_mask = batch['batch_inputs_pr']['attention_mask'].to(self.device)
                # labels = torch.tensor(batch['labels']).to(self.device)
                labels = torch.tensor(
                    batch['labels'],
                    dtype=torch.float32 if self.task_mode == "regression" else torch.long,
                ).to(self.device)
                regression_targets = self.target_scaler.transform(labels) if self.task_mode == "regression" else None

                drug_labels = batch['masked_drug_labels']
                drug_flex = batch.get('batch_drug_flex')
                if drug_flex is not None:
                    drug_flex = drug_flex.to(self.device)
                protein_flex = batch.get('batch_protein_flex')
                if protein_flex is not None:
                    protein_flex = protein_flex.to(self.device)
                mlm_loss = labels.new_tensor(0.0, dtype=torch.float32)
                if drug_labels is not None:
                    masked_drugs = batch['batch_inputs_drug_m'].to(self.device)
                    drug_labels = drug_labels.to(self.device)
                    output = self.model(
                        input_drugs,
                        input_proteins,
                        pr_mask=pr_mask,
                        masked_drugs=masked_drugs,
                        drug_flex=drug_flex,
                        protein_flex=protein_flex,
                        protein_ids=tuple(batch["protein_ids"]),
                    )
                    if (drug_labels != -1).any():
                        mlm_loss = nn.CrossEntropyLoss(ignore_index=-1)(
                            output['drug_mlm_logits'], drug_labels
                        )
                else:
                    output = self.model(
                        input_drugs,
                        input_proteins,
                        pr_mask=pr_mask,
                        drug_flex=drug_flex,
                        protein_flex=protein_flex,
                        protein_ids=tuple(batch["protein_ids"]),
                    )

                if self.task_mode == "regression":
                    dti_loss = self.criterion(output['prediction'], regression_targets)
                else:
                    dti_loss = self.criterion(output['logits'], labels)
                task_loss = dti_loss + self.lambda_mlm * mlm_loss
                loss = self._add_auxiliary_losses(output, task_loss, labels=labels)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss at "
                    f"epoch={self.current_epoch}, step={step}: "
                    f"dti={self._loss_item(dti_loss):.6f}, "
                    f"mlm={self._loss_item(mlm_loss):.6f}, "
                    f"relation={self._loss_item(output.get('relation_loss')):.6f}"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(self.optim)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )
            if not torch.isfinite(grad_norm):
                nonfinite_gradients = [
                    name for name, parameter in self.model.named_parameters()
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                ]
                raise FloatingPointError(
                    "Non-finite gradient norm at "
                    f"epoch={self.current_epoch}, step={step}; "
                    f"nonfinite_parameters={nonfinite_gradients}"
                )
            scaler.step(self.optim)
            scaler.update()

            if self.structural_enabled and (step + 1) % self.structural_interval == 0:
                try:
                    structural_batch = next(structural_iterator)
                except StopIteration:
                    structural_iterator = iter(self.structural_dataloader)
                    structural_batch = next(structural_iterator)
                structural_loss = self._run_structural_step(structural_batch, scaler)
                structural_loss_epoch += float(structural_loss.detach())
                structural_semantic_epoch += getattr(self, "_last_structural_semantic_loss", 0.0)
                structural_residue_rank_epoch += getattr(self, "_last_structural_residue_ranking_loss", 0.0)
                structural_residue_rank_pairs += getattr(self, "_last_structural_residue_ranking_pair_count", 0)
                structural_residue_rank_skipped += getattr(self, "_last_structural_residue_ranking_skipped_sample_count", 0)
                structural_steps += 1


            loss_epoch += loss.item()
            relation_loss_epoch += self._loss_item(output.get('relation_loss'))
            drug_relation_loss_epoch += self._loss_item(output.get('drug_relation_loss'))
            protein_relation_loss_epoch += self._loss_item(output.get('protein_relation_loss'))
            positive_count_epoch += self._loss_item(output.get('relation_positive_count'))
            valid_relation_batches += int(bool(output.get('valid_relation_batch', False)))
            drug_cos_epoch += self._loss_item(output.get('drug_response_offdiag_cos'))
            protein_cos_epoch += self._loss_item(output.get('protein_response_offdiag_cos'))
            response_std_epoch += self._loss_item(output.get('response_mean_std'))
            if not self.quiet_progress:
                loop.set_postfix(
                    avg_loss=loss_epoch / (step + 1),
                    dti_loss=self._loss_item(dti_loss),
                    mlm_loss=self._loss_item(mlm_loss),
                    relation_loss=self._loss_item(output.get('relation_loss')),
                    total_loss=self._loss_item(loss),
                    grad_norm=self._loss_item(grad_norm),
                )

        n_batches = max(len(self.train_dataloader), 1)
        if self.alignment_target != "none":
            print(
                f"Epoch {self.current_epoch:03d} relation | "
                f"relation_loss={relation_loss_epoch / n_batches:.4f} "
                f"drug_rel={drug_relation_loss_epoch / n_batches:.4f} "
                f"protein_rel={protein_relation_loss_epoch / n_batches:.4f} "
                f"pos/batch={positive_count_epoch / n_batches:.2f} "
                f"valid_batch_ratio={valid_relation_batches / n_batches:.3f} "
                f"drug_cos={drug_cos_epoch / n_batches:.4f} "
                f"protein_cos={protein_cos_epoch / n_batches:.4f} "
                f"resp_std={response_std_epoch / n_batches:.4f}",
                flush=True,
            )

        if structural_steps:
            print(
                f"Epoch {self.current_epoch:03d} structure | "
                f"weighted_response_loss={structural_loss_epoch / structural_steps:.4f} "
                f"semantic_loss={structural_semantic_epoch / structural_steps:.4f} "
                f"residue_rank_loss={structural_residue_rank_epoch / structural_steps:.4f} "
                f"residue_rank_pairs={structural_residue_rank_pairs} "
                f"residue_rank_skipped_samples={structural_residue_rank_skipped} "
                f"steps={structural_steps}",
                flush=True,
            )

        return loss_epoch / len(self.train_dataloader)

    def _normalize_structural_components(self, geometry, semantic, semantic_weight, geometry_weight=1.0):
        """Return raw or EMA-normalized geometry / semantic structural losses.

        EMA values are detached running scales. They only balance loss magnitudes;
        they are never differentiated and do not alter the default ``none`` path.
        Other optional structural terms remain in their original units.
        """
        if getattr(self, "structural_loss_normalization", "none") == "none":
            components = {
                "geometry": float(geometry_weight) * geometry,
                "semantic": float(semantic_weight) * semantic,
            }
            return components["geometry"] + components["semantic"], components

        def normalized(name, value):
            if not bool(torch.isfinite(value.detach())):
                raise FloatingPointError(f"Non-finite structural {name} loss")
            value_detached = value.detach()
            if not hasattr(self, "_structural_loss_ema"):
                self._structural_loss_ema = {}
            previous = self._structural_loss_ema.get(name)
            decay = getattr(self, "structural_loss_ema_decay", 0.99)
            scale = value_detached if previous is None else (
                decay * previous + (1.0 - decay) * value_detached
            )
            self._structural_loss_ema[name] = scale
            return value / scale.clamp_min(torch.finfo(value.dtype).eps)

        components = {
            "geometry": (
                float(geometry_weight) * normalized("geometry", geometry)
                if float(geometry_weight) > 0.0 else geometry * 0.0
            )
        }
        if float(semantic_weight) > 0.0:
            components["semantic"] = float(semantic_weight) * normalized(
                "semantic", semantic
            )
        else:
            components["semantic"] = semantic * 0.0
        return components["geometry"] + components["semantic"], components

    def _run_structural_step(self, batch, scaler):
        """One target-only structural calibration update after a DTI update."""
        self.optim.zero_grad()
        with autocast(device_type="cuda", enabled=self.use_amp):
            response_loss = structural_response_loss_from_batch(
                self.model,
                batch,
                self.device,
                margin=self.structural_margin,
                label_perturbation=self.structural_label_perturbation,
                label_perturbation_seed=self.structural_label_perturbation_seed + self.step,
                ranking_weight=self.structural_ranking_weight,
                target_random_deletion_ratio=getattr(
                    self, "structural_target_random_deletion_ratio", 0.0
                ),
                mode=getattr(self, "structural_mode", "scalar_rmsd"),
                gearnet_weight=getattr(self, "structural_gearnet_weight", 0.0),
                gearnet_semantic_weight=getattr(self, "structural_gearnet_semantic_weight", 0.0),
                gearnet_global_weight=getattr(self, "structural_gearnet_global_weight", 0.0),
                gearnet_relational_weight=getattr(self, "structural_gearnet_relational_weight", 0.0),
                gearnet_relational_temperature=getattr(self, "structural_gearnet_relational_temperature", 0.1),
                gearnet_direct_weight=getattr(self, "structural_gearnet_direct_weight", 0.0),
                gearnet_pair_weight=getattr(self, "structural_gearnet_pair_weight", 0.0),
                gearnet_pair_temperature=getattr(self, "structural_gearnet_pair_temperature", 0.5),
                site_ranking_weight=getattr(self, "structural_site_ranking_weight", 0.0),
                site_ranking_min_gap=getattr(self, "structural_site_ranking_min_gap", 0.1),
                residue_ranking_weight=getattr(self, "structural_residue_ranking_weight", 0.0),
                residue_ranking_min_gap=getattr(self, "structural_residue_ranking_min_gap", 0.10),
                residue_ranking_temperature=getattr(self, "structural_residue_ranking_temperature", 0.20),
                contact_weight_mode=getattr(self, "structural_contact_weight_mode", "none"),
                contact_weight_alpha=getattr(self, "structural_contact_weight_alpha", 1.0),
                contact_weight_sigma=getattr(self, "structural_contact_weight_sigma", 4.5),
                contact_weight_shuffle_seed=getattr(self, "structural_contact_weight_shuffle_seed", 0),
            )
            semantic_weight = (
                getattr(self, "structural_gearnet_semantic_weight", 0.0)
                + getattr(self, "structural_gearnet_global_weight", 0.0)
                + getattr(self, "structural_gearnet_relational_weight", 0.0)
                + getattr(self, "structural_gearnet_direct_weight", 0.0)
                + getattr(self, "structural_gearnet_pair_weight", 0.0)
            )
            normalized_total, normalized_components = self._normalize_structural_components(
                geometry=response_loss.magnitude,
                semantic=response_loss.semantic,
                semantic_weight=semantic_weight,
                geometry_weight=getattr(self, "structural_geometry_weight", 1.0),
            )
            raw_normalized_terms = (
                response_loss.magnitude + semantic_weight * response_loss.semantic
            )
            # Preserve optional ranking / teacher-magnitude terms in their original
            # scale. The EMA mode only rebalances the current geometry and semantic
            # terms, which are the two components used by the semantic P2 run.
            other_terms = response_loss.total - raw_normalized_terms
            loss = self.structural_weight * (normalized_total + other_terms)
        self._last_structural_semantic_loss = float(response_loss.semantic.detach())
        self._last_structural_residue_ranking_loss = float((response_loss.residue_ranking if response_loss.residue_ranking is not None else response_loss.magnitude * 0.0).detach())
        self._last_structural_residue_ranking_pair_count = int(response_loss.residue_ranking_pair_count)
        self._last_structural_residue_ranking_skipped_sample_count = int(response_loss.residue_ranking_skipped_sample_count)
        self._last_structural_normalized_geometry_loss = float(
            normalized_components["geometry"].detach()
        )
        self._last_structural_normalized_semantic_loss = float(
            normalized_components["semantic"].detach()
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite structural response loss")
        scaler.scale(loss).backward()
        scaler.unscale_(self.optim)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.grad_clip
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite structural response gradient norm")
        scaler.step(self.optim)
        scaler.update()
        return loss.detach()

    def _compute_weighted_if_score(self, if_tensor, delta_tensor, mask):
        eps = 1e-8
        if if_tensor is None or delta_tensor is None:
            batch_size = mask.size(0)
            return torch.zeros(batch_size, device=mask.device, dtype=torch.float32)
        if_values = if_tensor.squeeze(-1).float() * mask.float()
        delta_norm = torch.linalg.vector_norm(delta_tensor.float(), ord=2, dim=-1)
        numerator = (if_values * delta_norm).sum(dim=1)
        denominator = if_values.sum(dim=1).clamp(min=eps)
        return numerator / denominator

    def _protein_flex_mean(self, protein_flex, protein_mask):
        if protein_flex is None:
            return None
        values = protein_flex.squeeze(-1).float() * protein_mask.float()
        denom = protein_mask.float().sum(dim=1).clamp(min=1.0)
        return values.sum(dim=1) / denom

    def _if_score_output_path(self, dataloader):
        if self.save_if_score_path:
            base = self.save_if_score_path
        else:
            base = os.path.join(self.output_dir, "if_scores.csv")
        root, ext = os.path.splitext(base)
        if ext.lower() != ".csv":
            os.makedirs(base, exist_ok=True)
            base = os.path.join(base, "if_scores.csv")
            root, ext = os.path.splitext(base)
        if dataloader == "val":
            return f"{root}_val_epoch{self.current_epoch}{ext}"
        return f"{root}_test{ext}"

    def _save_if_scores(self, rows, dataloader):
        if not rows:
            return
        save_path = self._if_score_output_path(dataloader)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(save_path, index=False)
        print(f"[Saved] induced flexibility scores saved to: {save_path}")

    def _build_if_score_rows(
        self,
        batch,
        output,
        scores,
        labels,
        drug_flex,
        protein_flex,
        protein_mask,
    ):
        protein_ids = [str(x) for x in batch.get('protein_ids', [''] * len(labels))]
        drug_ids = [str(x) for x in batch.get('drug_ids', [''] * len(labels))]
        if self.target_protein_for_case_study:
            keep = [pid == self.target_protein_for_case_study for pid in protein_ids]
        else:
            keep = [True] * len(protein_ids)

        protein_score = self._compute_weighted_if_score(
            output.get('protein_induced_if'),
            output.get('protein_induced_delta'),
            protein_mask,
        )
        drug_mask = batch['batch_inputs_drug']['attention_mask'].to(self.device)
        drug_score = self._compute_weighted_if_score(
            output.get('drug_induced_if'),
            output.get('drug_induced_delta'),
            drug_mask,
        )
        total_score = self.if_score_alpha * protein_score + (1.0 - self.if_score_alpha) * drug_score

        protein_flex_mean = self._protein_flex_mean(protein_flex, protein_mask)
        if drug_flex is not None:
            drug_flex_values = drug_flex[:, 0].detach().cpu().tolist()
        else:
            drug_flex_values = [np.nan] * len(protein_ids)
        if protein_flex_mean is not None:
            protein_flex_mean_values = protein_flex_mean.detach().cpu().tolist()
        else:
            protein_flex_mean_values = [np.nan] * len(protein_ids)

        rows = []
        labels_list = [int(x) for x in labels]
        pred_list = scores.detach().cpu().tolist()
        sp_list = protein_score.detach().cpu().tolist()
        sd_list = drug_score.detach().cpu().tolist()
        st_list = total_score.detach().cpu().tolist()
        for i, should_keep in enumerate(keep):
            if not should_keep:
                continue
            rows.append({
                'protein_id': protein_ids[i],
                'drug_id': drug_ids[i],
                'label': labels_list[i],
                'pred_score': pred_list[i],
                'S_p_IF': sp_list[i],
                'S_d_IF': sd_list[i],
                'S_IF': st_list[i],
                'drug_flex': drug_flex_values[i],
                'protein_flex_mean': protein_flex_mean_values[i],
            })
        return rows

    def test(self, dataloader="test", _return_threshold=False):
        # Fit on validation predictions from this exact model state, including
        # checkpoint-only evaluation. Never reuse a threshold from another epoch.
        optimal_threshold = None
        if self.task_mode != "regression" and dataloader == "test":
            optimal_threshold = self.test(dataloader="val", _return_threshold=True)
        y_label, y_pred = [], []
        if_score_rows = []
        prediction_rows = []
        loader = self.test_dataloader if dataloader == "test" else self.val_dataloader
        loop = tqdm(loader, file=sys.stdout, colour="#bdbdbd", disable=self.quiet_progress, leave=False)

        with torch.no_grad():
            self.model.eval()
            if not self.quiet_progress:
                loop.set_description("Test" if dataloader == "test" else "Validation")
            for batch in loop:
                labels = batch['labels']
                input_proteins = batch['batch_inputs_pr']['input_ids'].to(self.device)
                input_drugs = batch['batch_inputs_drug'].to(self.device)
                pr_mask = batch['batch_inputs_pr']['attention_mask'].to(self.device)
                drug_flex = batch.get('batch_drug_flex')
                if drug_flex is not None:
                    drug_flex = drug_flex.to(self.device)
                protein_flex = batch.get('batch_protein_flex')
                if protein_flex is not None:
                    protein_flex = protein_flex.to(self.device)
                output = self.model(
                    input_drugs, input_proteins, pr_mask=pr_mask,
                    drug_flex=drug_flex, protein_flex=protein_flex,
                    protein_ids=tuple(batch["protein_ids"]),
                )
                if self.task_mode == "regression":
                    scores = self.target_scaler.inverse(output['prediction'])
                    y_label.extend(float(value) for value in labels)
                    y_pred.extend(scores.detach().cpu().tolist())
                    continue

                logits = output['logits']
                scores = torch.softmax(logits, dim=-1)[:, 1]
                if dataloader == "test" and self.prediction_export_path:
                    for protein_id, drug_id, label, score in zip(
                        batch["protein_ids"], batch["drug_ids"], labels,
                        scores.detach().cpu().tolist(),
                    ):
                        prediction_rows.append({
                            "row_index": len(prediction_rows),
                            "protein_id": str(protein_id),
                            "drug_id": str(drug_id),
                            "label": int(label),
                            "pred_score": float(score),
                        })
                if self.compute_if_score:
                    if_score_rows.extend(self._build_if_score_rows(
                        batch=batch, output=output, scores=scores, labels=labels,
                        drug_flex=drug_flex, protein_flex=protein_flex, protein_mask=pr_mask,
                    ))
                y_label.extend(labels)
                y_pred.extend(scores.detach().cpu().tolist())

        if self.task_mode == "regression":
            return regression_metrics(y_label, y_pred)
        if self.compute_if_score:
            self._save_if_scores(if_score_rows, dataloader)
        if dataloader == "test" and self.prediction_export_path:
            path = Path(self.prediction_export_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(prediction_rows).to_csv(path, index=False)
            print(f"[Saved] per-sample predictions: {path}", flush=True)

        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        if dataloader != "test":
            if _return_threshold:
                fpr, tpr, thresholds = roc_curve(y_label, y_pred)
                # ROC may include an infinite/supra-probability sentinel. Keep
                # only realizable probability thresholds, with stable first tie.
                valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
                if not valid.any() or len(np.unique(y_label)) != 2:
                    raise ValueError("Threshold selection requires both validation classes and finite probability thresholds.")
                candidates = np.flatnonzero(valid)
                return float(thresholds[candidates[np.argmax((tpr - fpr)[valid])]])
            return auroc, auprc
        y_bin = (np.array(y_pred) >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_label, y_bin).ravel()
        return (
            auroc, auprc, f1_score(y_label, y_bin, zero_division=0), precision_score(y_label, y_bin, zero_division=0),
            tp / (tp + fn), tn / (tn + fp), accuracy_score(y_label, y_bin),
            matthews_corrcoef(y_label, y_bin), optimal_threshold,
        )
