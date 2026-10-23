"""Training, evaluation, and atomic checkpointing for conformation pretraining."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .objectives import compute_conformation_loss, evaluate_conformation_predictions


@dataclass(frozen=True)
class FitResult:
    best_checkpoint: Path
    best_epoch: int
    best_valid_loss: float
    history: tuple[dict, ...]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ConformationTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        output_dir,
        model_config,
        training_config,
        filtered_sample_hash,
        feature_manifest_hash,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        self.model_config = model_config
        self.training_config = training_config
        self.filtered_sample_hash = filtered_sample_hash
        self.feature_manifest_hash = feature_manifest_hash
        self.epochs = int(training_config.get("epochs", 50))
        self.patience = int(training_config.get("patience", 10))
        self.gradient_clip = float(training_config.get("gradient_clip", 1.0))
        self.pocket_weight = float(training_config.get("pocket_weight", 1.0))
        self.global_weight = float(training_config.get("global_weight", 0.2))
        self.amp_enabled = bool(training_config.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

    def _loss(self, output, targets):
        return compute_conformation_loss(
            output, targets,
            pocket_weight=self.pocket_weight,
            global_weight=self.global_weight,
        )

    @staticmethod
    def _averages(sums, batches):
        denominator = max(int(batches), 1)
        return {key: value / denominator for key, value in sums.items()}

    def train_epoch(self, loader):
        self.model.train()
        sums = {"total": 0.0, "all_residue": 0.0, "pocket": 0.0, "global": 0.0}
        batches = 0
        gradient_norms = []
        skipped_amp_steps = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                output = self.model(**batch.model_inputs)
                losses = self._loss(output, batch.targets)
            if not torch.isfinite(losses.total):
                raise FloatingPointError("Non-finite conformation training loss")
            self.scaler.scale(losses.total).backward()
            self.scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            gradient_is_finite = bool(torch.isfinite(torch.as_tensor(gradient_norm)))
            if not gradient_is_finite and not self.amp_enabled:
                raise FloatingPointError("Non-finite conformation gradient norm")
            if gradient_is_finite:
                gradient_norms.append(float(gradient_norm))
            else:
                # GradScaler recorded found_inf during unscale_ and will skip
                # this optimizer step, then reduce its scale in update().
                skipped_amp_steps += 1
            self.scaler.step(self.optimizer)
            self.scaler.update()
            sums["total"] += float(losses.total.detach())
            sums["all_residue"] += float(losses.all_residue.detach())
            sums["pocket"] += float(losses.pocket.detach())
            sums["global"] += float(losses.global_loss.detach())
            batches += 1
        result = self._averages(sums, batches)
        result["gradient_norm"] = float(np.mean(gradient_norms)) if gradient_norms else float("nan")
        result["skipped_amp_steps"] = int(skipped_amp_steps)
        return result

    def evaluate(self, loader, split="valid", write_outputs=True):
        self.model.eval()
        sums = {"total": 0.0, "all_residue": 0.0, "pocket": 0.0, "global": 0.0}
        batches = 0
        records = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    output = self.model(**batch.model_inputs)
                    losses = self._loss(output, batch.targets)
                if not torch.isfinite(losses.total):
                    raise FloatingPointError(f"Non-finite {split} loss")
                sums["total"] += float(losses.total)
                sums["all_residue"] += float(losses.all_residue)
                sums["pocket"] += float(losses.pocket)
                sums["global"] += float(losses.global_loss)
                batches += 1
                prediction = output.log_displacement.detach().float().cpu()
                target = batch.targets["log_displacement"].detach().float().cpu()
                valid = batch.targets["valid_mask"].detach().bool().cpu()
                pocket = batch.targets["pocket_mask"].detach().bool().cpu() & valid
                global_prediction = output.log_global_targets.detach().float().cpu()
                global_target = batch.targets["log_global_targets"].detach().float().cpu()
                for row_index, sample_id in enumerate(batch.sample_ids):
                    for position in torch.nonzero(valid[row_index], as_tuple=False).flatten().tolist():
                        base = {
                            "sample_id": sample_id,
                            "position": int(position),
                            "log_prediction": float(prediction[row_index, position]),
                            "log_target": float(target[row_index, position]),
                        }
                        records.append({**base, "scope": "residue_all"})
                        if bool(pocket[row_index, position]):
                            records.append({**base, "scope": "residue_pocket"})
                    records.append({
                        "sample_id": sample_id, "position": -1, "scope": "global_mean",
                        "log_prediction": float(global_prediction[row_index, 0]),
                        "log_target": float(global_target[row_index, 0]),
                    })
                    records.append({
                        "sample_id": sample_id, "position": -1, "scope": "global_max",
                        "log_prediction": float(global_prediction[row_index, 1]),
                        "log_target": float(global_target[row_index, 1]),
                    })
        result = {"loss": self._averages(sums, batches), "metrics": evaluate_conformation_predictions(records)}
        if write_outputs:
            pd.DataFrame(records).to_csv(self.output_dir / f"{split}_predictions.csv", index=False)
            (self.output_dir / f"{split}_metrics.json").write_text(
                json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
            )
        return result

    def _checkpoint_payload(self, epoch, valid_loss):
        return {
            "format_version": 1,
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model_config,
            "training_config": self.training_config,
            "filtered_sample_hash": self.filtered_sample_hash,
            "feature_manifest_hash": self.feature_manifest_hash,
            "best_epoch": int(epoch),
            "best_valid_loss": float(valid_loss),
        }

    def _save_checkpoint(self, path, epoch, valid_loss):
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self._checkpoint_payload(epoch, valid_loss), temporary)
        os.replace(temporary, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if int(checkpoint.get("format_version", -1)) != 1:
            raise ValueError("Unsupported conformation checkpoint format")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return checkpoint

    def fit(self, train_loader, valid_loader):
        history = []
        best_loss = float("inf")
        best_epoch = -1
        stale_epochs = 0
        best_path = self.output_dir / "checkpoints" / "best.pt"
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            valid = self.evaluate(valid_loader, split="valid", write_outputs=False)
            valid_loss = float(valid["loss"]["total"])
            row = {"epoch": epoch}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"valid_{key}": value for key, value in valid["loss"].items()})
            history.append(row)
            pd.DataFrame(history).to_csv(self.output_dir / "history.csv", index=False)
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                stale_epochs = 0
                self._save_checkpoint(best_path, epoch, valid_loss)
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        if best_epoch < 0:
            raise RuntimeError("Training produced no valid checkpoint")
        self.load_checkpoint(best_path)
        manifest = {
            "best_epoch": best_epoch,
            "best_valid_loss": best_loss,
            "epochs_completed": len(history),
            "filtered_sample_hash": self.filtered_sample_hash,
            "feature_manifest_hash": self.feature_manifest_hash,
        }
        (self.output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return FitResult(best_path, best_epoch, best_loss, tuple(history))
