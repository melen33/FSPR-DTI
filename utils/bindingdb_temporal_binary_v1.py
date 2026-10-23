"""Build the isolated BindingDB temporal binary IC50 benchmark (protocol v1)."""

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem


_AA = set("ACDEFGHIKLMNPQRSTVWY")
_IC50 = re.compile(r"^\s*(<=|>=|<|>|=)?\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*$")
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
_CONFLICT_COLUMNS = ["pair_id", "measurement_count", "labels", "record_ids"]


def _get(row, columns, *names):
    for name in names:
        index = columns.get(name)
        if index is not None and index < len(row):
            return row[index].strip()
    return ""


def _parse_ic50(text):
    match = _IC50.match(text or "")
    return (float(match.group(2)), match.group(1) or "=") if match else (None, None)


def _label(value, operator):
    if operator in ("=", "<", "<=") and value <= 100.0:
        return 1
    if operator in ("=", ">", ">=") and value >= 10000.0:
        return 0
    return None


def _normalize_smiles(value):
    molecule = Chem.MolFromSmiles(value)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True) if molecule else None


def _normalize_sequence(value):
    sequence = "".join((value or "").upper().split())
    return sequence if sequence and set(sequence).issubset(_AA) else None


def _parse_year(value):
    match = _YEAR.search(value or "")
    return int(match.group(1)) if match else None


def _pair_id(smiles, sequence):
    return hashlib.sha256((smiles + "\n" + sequence).encode("utf-8")).hexdigest()


def _atomic_csv(frame, path, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.reindex(columns=columns).to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(input_tsv):
    counts, kept = Counter(), []
    with Path(input_tsv).open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        columns = {name.strip(): index for index, name in enumerate(header)}
        required = ["Ligand SMILES", "IC50 (nM)", "Date of publication", "BindingDB Target Chain Sequence 1"]
        missing = [name for name in required if name not in columns]
        if missing:
            raise ValueError("TSV missing required columns: " + ", ".join(missing))
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
            smiles = _normalize_smiles(_get(row, columns, "Ligand SMILES"))
            if smiles is None:
                counts["invalid_smiles"] += 1
                continue
            protein = _normalize_sequence(_get(row, columns, "BindingDB Target Chain Sequence 1"))
            if protein is None:
                counts["invalid_protein"] += 1
                continue
            patent = _get(row, columns, "Patent Number")
            doi, pmid = _get(row, columns, "Article DOI"), _get(row, columns, "PMID")
            source_kind = "patent" if patent else "publication" if (doi or pmid) else "unattributed"
            if source_kind == "unattributed":
                counts["unattributed_source"] += 1
                continue
            kept.append({
                "pair_id": _pair_id(smiles, protein), "SMILES": smiles, "Protein": protein, "Y": label,
                "Year": year, "record_id": _get(row, columns, "BindingDB Reactant_set_id") or str(ordinal),
                "source_kind": source_kind, "patent_number": patent, "article_doi": doi, "pmid": pmid,
                "uniprot_id": _get(row, columns, "UniProt (SwissProt) Primary ID of Target Chain 1"),
            })
            counts["accepted_measurements"] += 1
            counts["positive_measurements" if label else "negative_measurements"] += 1
            counts[source_kind + "_measurements"] += 1
    return pd.DataFrame(kept), counts


def _aggregate(records):
    pairs, conflicts, historical_without_eligible = [], [], set()
    for pair_id, group in records.groupby("pair_id", sort=True):
        labels = sorted(group["Y"].astype(int).unique().tolist())
        if len(labels) != 1:
            conflicts.append({"pair_id": pair_id, "measurement_count": int(len(group)), "labels": ";".join(map(str, labels)), "record_ids": ";".join(group["record_id"].astype(str))})
            continue
        eligible = group.loc[group["Year"] >= 2013]
        if eligible.empty:
            historical_without_eligible.add(pair_id)
            continue
        anchor = eligible.loc[eligible["Year"].idxmin()]
        pairs.append({
            "pair_id": pair_id, "SMILES": anchor["SMILES"], "Protein": anchor["Protein"], "Y": labels[0],
            "Year": int(eligible["Year"].min()), "first_any_year": int(group["Year"].min()),
            "measurement_count": int(len(group)), "source_kinds": ";".join(sorted(group["source_kind"].unique().tolist())),
            "uniprot_id": anchor["uniprot_id"], "record_ids": ";".join(group["record_id"].astype(str)),
        })
    return pd.DataFrame(pairs), pd.DataFrame(conflicts, columns=_CONFLICT_COLUMNS), historical_without_eligible


def _split(pairs):
    masks = {
        "train": (pairs.Year >= 2013) & (pairs.Year <= 2019),
        "validation": (pairs.Year >= 2020) & (pairs.Year <= 2021),
        "test": (pairs.Year >= 2022) & (pairs.Year <= 2025) & (pairs.first_any_year >= 2022),
        "prospective_2026": (pairs.Year == 2026) & (pairs.first_any_year >= 2026),
    }
    return {name: pairs.loc[mask].sort_values("pair_id").reset_index(drop=True) for name, mask in masks.items()}


def build_benchmark(input_tsv, output_root, dry_run=False):
    """Build v1 outputs, or return a no-write measurement audit in dry-run mode."""
    input_tsv, output_root = Path(input_tsv), Path(output_root)
    records, counts = _records(input_tsv)
    measurement_audit = {"input_tsv": str(input_tsv), "input_sha256": _digest(input_tsv), "counts": dict(sorted(counts.items()))}
    if dry_run:
        return {"dry_run": True, "measurement_audit": measurement_audit}
    if records.empty:
        raise ValueError("no valid, attributable, binary IC50 records remain")
    pairs, conflicts, historical_without_eligible = _aggregate(records)
    if pairs.empty:
        raise ValueError("no consistent pair remains after aggregation")
    splits = _split(pairs)
    split_root, audit_root = output_root / "splits", output_root / "audit"
    split_paths, statistics = {}, {}
    metadata_columns = list(pairs.columns)
    for name, frame in splits.items():
        model_path, metadata_path = split_root / (name + ".csv"), split_root / (name + "_metadata.csv")
        _atomic_csv(frame, model_path, ["SMILES", "Protein", "Y"])
        _atomic_csv(frame, metadata_path, metadata_columns)
        split_paths[name] = model_path
        statistics[name] = {"rows": int(len(frame)), "positive": int(frame.Y.sum()), "negative": int((frame.Y == 0).sum()), "positive_ratio": float(frame.Y.mean()) if len(frame) else None, "unique_drugs": int(frame.SMILES.nunique()), "unique_proteins": int(frame.Protein.nunique()), "unique_pairs": int(frame.pair_id.nunique()), "year_min": int(frame.Year.min()) if len(frame) else None, "year_max": int(frame.Year.max()) if len(frame) else None}
    names = list(splits)
    overlaps = {a + "__" + b: int(len(set(splits[a].pair_id) & set(splits[b].pair_id))) for index, a in enumerate(names) for b in names[index + 1:]}
    leakage = {"pair_overlap_all_zero": not any(overlaps.values()), "pair_overlap_counts": overlaps, "historical_pairs_without_eligible_rows": len(historical_without_eligible), "historical_pairs_blocked_from_later_splits": int((pairs.first_any_year < pairs.Year).sum()), "test_pairs_with_pre_2022_history": int((splits["test"].first_any_year < 2022).sum()), "prospective_pairs_with_pre_2026_history": int((splits["prospective_2026"].first_any_year < 2026).sum())}
    _atomic_csv(conflicts, audit_root / "pair_conflicts.csv", _CONFLICT_COLUMNS)
    _atomic_json(measurement_audit, audit_root / "row_filter_audit.json")
    _atomic_json(statistics, audit_root / "split_statistics.json")
    _atomic_json(leakage, audit_root / "leakage_audit.json")
    _atomic_json({"protocol_version": "bindingdb_temporal_binary_v1", "source_sha256": measurement_audit["input_sha256"], "statistics": statistics, "leakage": leakage}, audit_root / "build_manifest.json")
    (output_root / "DATA_CARD.md").write_text("# BindingDB temporal binary v1\n\nPositive: IC50 <=100 nM. Negative: IC50 >=10,000 nM. Gray-zone and conflicting pairs are removed. Train: 2013--2019; validation: 2020--2021; main test: 2022--2025; 2026 is prospective supplementary only. Model CSVs contain exactly `SMILES,Protein,Y`; full provenance is in metadata and audit files.\n", encoding="utf-8")
    return {"root": output_root, "splits": split_paths, "statistics": statistics, "leakage": leakage}
