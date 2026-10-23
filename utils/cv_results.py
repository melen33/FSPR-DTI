from pathlib import Path
from numbers import Real


def numeric_metric_keys(result):
    """Return numeric result fields suitable for mean/std CV aggregation."""
    return [
        key for key, value in result.items()
        if key != "best_epoch" and isinstance(value, Real) and not isinstance(value, bool)
    ]


def resolve_cv_results_path(args, config, data_name, split_name):
    """Return an isolated CV summary path while preserving the legacy default."""
    explicit = getattr(args, "cv_results_file", None)
    if explicit:
        return Path(explicit)
    return (
        Path("results")
        / data_name
        / split_name
        / f"cv_results_seed{config.TRAIN.SEED}.txt"
    )

