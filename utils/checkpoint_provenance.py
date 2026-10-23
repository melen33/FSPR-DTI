"""Helpers for recording checkpoint-affecting DTI model provenance."""


def record_conformation_pretraining_provenance(model_config, checkpoint_path):
    """Persist the requested transfer path and exact transfer-loaded boolean."""
    normalized_path = "" if checkpoint_path is None else str(checkpoint_path)
    model_config["conformation_pretrained_checkpoint"] = normalized_path
    model_config["conformation_pretrained_loaded"] = bool(normalized_path)
    return model_config


def merge_with_conformation_pretraining_provenance(
    train_config, model_config, checkpoint_path, merge_fn
):
    """Record CLI transfer provenance before executing the supplied merge."""
    record_conformation_pretraining_provenance(model_config, checkpoint_path)
    return merge_fn(train_config, model_config)

