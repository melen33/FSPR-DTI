"""Constant-memory schema and measurement audit for official BindingDB TSV files."""

import csv
import hashlib
from collections import Counter
from pathlib import Path

from rdkit import Chem

from utils.bindingdb_temporal_binary_v1 import _label, _parse_ic50, _parse_year, _normalize_sequence


def _get(row, columns, *names):
    for name in names:
        index = columns.get(name)
        if index is not None and index < len(row):
            return row[index].strip()
    return ""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_tsv(input_tsv, progress_every=250000):
    """Return aggregate audit counts while retaining no per-measurement records."""
    input_tsv = Path(input_tsv)
    counts, years = Counter(), Counter()
    with input_tsv.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        columns = {name.strip(): index for index, name in enumerate(header)}
        required = ["Ligand SMILES", "IC50 (nM)", "Date of publication"]
        missing = [name for name in required if name not in columns]
        if missing:
            raise ValueError("TSV missing required columns: " + ", ".join(missing))
        sequence_names = ("BindingDB Target Chain Sequence", "BindingDB Target Chain Sequence 1")
        if not any(name in columns for name in sequence_names):
            raise ValueError("TSV missing required protein-chain sequence field")
        for ordinal, row in enumerate(reader, 1):
            counts["raw_rows"] += 1
            value, operator = _parse_ic50(_get(row, columns, "IC50 (nM)"))
            if value is None:
                counts["invalid_ic50"] += 1
                continue
            label = _label(value, operator)
            if label is None:
                counts["gray_or_ambiguous_ic50"] += 1
                continue
            year = _parse_year(_get(row, columns, "Date of publication"))
            if year is None:
                counts["invalid_year"] += 1
                continue
            molecule = Chem.MolFromSmiles(_get(row, columns, "Ligand SMILES"))
            if molecule is None:
                counts["invalid_smiles"] += 1
                continue
            if _normalize_sequence(_get(row, columns, *sequence_names)) is None:
                counts["invalid_protein"] += 1
                continue
            patent = _get(row, columns, "Patent Number")
            article = _get(row, columns, "Article DOI") or _get(row, columns, "PMID")
            if not patent and not article:
                counts["unattributed_source"] += 1
                continue
            counts["accepted_measurements"] += 1
            counts["positive_measurements" if label else "negative_measurements"] += 1
            counts["patent_measurements" if patent else "publication_measurements"] += 1
            years[str(year)] += 1
    return {"input_tsv": str(input_tsv), "input_sha256": _sha256(input_tsv), "counts": dict(sorted(counts.items())), "accepted_year_counts": dict(sorted(years.items()))}
