"""Paths and label-isolation rules for standardized cross-domain DTI data."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CrossDomainPaths:
    root: Path
    train: Path
    validation: Path
    target_train: Path
    test: Path

    @property
    def feature_csvs(self):
        """CSV files whose sequences may be encoded without using target labels."""
        return (self.train, self.validation, self.test)


def cross_domain_paths(root: Path) -> CrossDomainPaths:
    """Resolve the standardized source-to-target protocol and validate its files."""
    root = Path(root)
    paths = CrossDomainPaths(
        root=root,
        train=root / "source_train.csv",
        validation=root / "source_val.csv",
        target_train=root / "target_train.csv",
        test=root / "target_test.csv",
    )
    missing = [str(path) for path in (
        paths.train, paths.validation, paths.target_train, paths.test
    ) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cross-domain standardized split is incomplete: " + ", ".join(missing)
        )
    return paths
