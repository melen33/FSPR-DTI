import pickle
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ("drug_id", "DrugBank ID", "DrugBank_ID", "SMILES", "smiles")
VALUE_COLUMNS = ("drug_flex_value", "nTABS", "ntabs", "drug_flex", "flex")


def _as_flex_vector(value, flex_dim, log1p=True):
    """Return a float32 vector with shape [flex_dim]."""
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError("Empty drug flexibility value.")
    if array.size == 1 and flex_dim > 1:
        array = np.repeat(array, flex_dim)
    if array.size != flex_dim:
        raise ValueError(
            f"Expected drug flexibility dim={flex_dim}, got dim={array.size}."
        )
    if log1p:
        if np.any(array < 0):
            raise ValueError("log1p drug flexibility requires non-negative values.")
        array = np.log1p(array)
    return array.astype(np.float32)


def _frame_to_map(df, flex_dim, log1p):
    key_column = next((column for column in KEY_COLUMNS if column in df.columns), None)
    value_column = next((column for column in VALUE_COLUMNS if column in df.columns), None)
    if key_column is None or value_column is None:
        raise ValueError(
            "Drug flexibility table must contain one key column "
            f"{KEY_COLUMNS} and one value column {VALUE_COLUMNS}. "
            f"Available columns: {list(df.columns)}"
        )

    flex_map = {}
    for _, row in df.iterrows():
        key = str(row[key_column])
        value = row[value_column]
        flex_map[key] = _as_flex_vector(value, flex_dim, log1p=log1p)
    return flex_map


def load_drug_flex_map(path, flex_dim=1, log1p=True):
    """Load precomputed drug flexibility priors.

    Supported formats:
    - CSV/TSV: one key column among drug_id/DrugBank ID/SMILES and one value
      column among drug_flex_value/nTABS/ntabs/drug_flex/flex.
    - PKL: either a dict {key: scalar_or_vector} or a pandas DataFrame.
    - NPY/NPZ: a saved dict-like object {key: scalar_or_vector}.

    Returned values are numpy float32 arrays with shape [flex_dim]. If log1p is
    true, raw nTABS values are transformed as F_d = log(1 + nTABS_d).
    """
    if path is None or str(path).strip() == "":
        return None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Drug flexibility file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return _frame_to_map(pd.read_csv(path, sep=sep), flex_dim, log1p)

    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, pd.DataFrame):
            return _frame_to_map(obj, flex_dim, log1p)
        if isinstance(obj, dict):
            return {
                str(key): _as_flex_vector(value, flex_dim, log1p=log1p)
                for key, value in obj.items()
            }
        raise ValueError(f"Unsupported PKL drug flexibility object: {type(obj)}")

    if suffix == ".npy":
        obj = np.load(path, allow_pickle=True)
        try:
            source = obj.item()
        except ValueError as exc:
            raise ValueError("NPY drug flexibility file must contain a saved dict.") from exc
        if not isinstance(source, dict):
            raise ValueError("NPY drug flexibility file must contain a saved dict.")
        return {
            str(key): _as_flex_vector(value, flex_dim, log1p=log1p)
            for key, value in source.items()
        }

    if suffix == ".npz":
        obj = np.load(path, allow_pickle=True)
        if "drug_flex" in obj and isinstance(obj["drug_flex"].item(), dict):
            source = obj["drug_flex"].item()
        else:
            source = {key: obj[key] for key in obj.files}
        return {
            str(key): _as_flex_vector(value, flex_dim, log1p=log1p)
            for key, value in source.items()
        }

    raise ValueError(f"Unsupported drug flexibility file type: {path.suffix}")

PROTEIN_KEY_COLUMNS = ("protein_id", "pr_id", "Protein_ID", "target_id", "uniprot_id")
PROTEIN_INDEX_COLUMNS = ("residue_index", "residue_idx", "position", "pos", "idx")
PROTEIN_S2_COLUMNS = ("S2", "s2", "dynamine_s2", "order_parameter")
PROTEIN_FLEX_COLUMNS = ("protein_flex", "F_p", "fp", "flex", "flexibility")
PROTEIN_VECTOR_COLUMNS = (
    "S2_values", "s2_values", "dynamine_s2_values",
    "protein_flex_values", "F_p_values", "flex_values", "values",
)


def _parse_vector(value):
    """Parse a scalar/list/string into a 1-D float32 numpy array."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Empty protein flexibility vector string.")
        if text[0] in "[ (" and text[-1] in "] )":
            import ast
            value = ast.literal_eval(text)
        else:
            for sep in (",", ";", "|"):
                text = text.replace(sep, " ")
            value = [item for item in text.split() if item]
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError("Empty protein flexibility vector.")
    return array.astype(np.float32)


def _protein_vector_to_flex(values, values_are_s2=True):
    """Return residue-level F_p with shape [L_p, 1]."""
    array = _parse_vector(values)
    if values_are_s2:
        array = 1.0 - array
    return array.reshape(-1, 1).astype(np.float32)


def _protein_frame_to_map(df, values_are_s2=True):
    key_column = next((column for column in PROTEIN_KEY_COLUMNS if column in df.columns), None)
    if key_column is None:
        raise ValueError(
            "Protein flexibility table must contain one key column "
            f"{PROTEIN_KEY_COLUMNS}. Available columns: {list(df.columns)}"
        )

    vector_column = next((column for column in PROTEIN_VECTOR_COLUMNS if column in df.columns), None)
    if vector_column is not None:
        column_lower = vector_column.lower()
        vector_is_s2 = values_are_s2 and not (
            'flex' in column_lower or 'f_p' in column_lower or column_lower.startswith('fp')
        )
        return {
            str(row[key_column]): _protein_vector_to_flex(row[vector_column], vector_is_s2)
            for _, row in df.iterrows()
        }

    s2_column = next((column for column in PROTEIN_S2_COLUMNS if column in df.columns), None)
    flex_column = next((column for column in PROTEIN_FLEX_COLUMNS if column in df.columns), None)
    value_column = s2_column or flex_column
    if value_column is None:
        raise ValueError(
            "Protein flexibility table must contain either a vector column "
            f"{PROTEIN_VECTOR_COLUMNS} or residue-level value column "
            f"{PROTEIN_S2_COLUMNS + PROTEIN_FLEX_COLUMNS}. "
            f"Available columns: {list(df.columns)}"
        )

    index_column = next((column for column in PROTEIN_INDEX_COLUMNS if column in df.columns), None)
    flex_map = {}
    for key, group in df.groupby(key_column, sort=False):
        if index_column is not None:
            group = group.sort_values(index_column)
        values = group[value_column].to_numpy(dtype=np.float32)
        value_is_s2 = value_column == s2_column
        flex_map[str(key)] = _protein_vector_to_flex(values, values_are_s2=value_is_s2)
    return flex_map


def load_protein_flex_map(path, values_are_s2=True):
    """Load precomputed residue-level protein flexibility priors.

    Supported formats:
    - CSV/TSV long format: protein_id/pr_id + residue_index + S2/s2.
    - CSV/TSV vector format: protein_id/pr_id + S2_values, e.g. "0.82 0.76 ...".
    - PKL: either a dict {protein_id: residue_vector} or a pandas DataFrame.
    - NPY/NPZ: a saved dict-like object {protein_id: residue_vector}.

    Returned values are numpy float32 arrays with shape [L_p, 1]. For DynaMine
    S^2 inputs, values are converted to F_p = 1 - S^2. If the table uses a
    protein_flex/F_p/flex column, values are assumed to already be F_p.
    """
    if path is None or str(path).strip() == "":
        return None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Protein flexibility file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return _protein_frame_to_map(pd.read_csv(path, sep=sep), values_are_s2=values_are_s2)

    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, pd.DataFrame):
            return _protein_frame_to_map(obj, values_are_s2=values_are_s2)
        if isinstance(obj, dict):
            return {
                str(key): _protein_vector_to_flex(value, values_are_s2=values_are_s2)
                for key, value in obj.items()
            }
        raise ValueError(f"Unsupported PKL protein flexibility object: {type(obj)}")

    if suffix == ".npy":
        obj = np.load(path, allow_pickle=True)
        try:
            source = obj.item()
        except ValueError as exc:
            raise ValueError("NPY protein flexibility file must contain a saved dict.") from exc
        if not isinstance(source, dict):
            raise ValueError("NPY protein flexibility file must contain a saved dict.")
        return {
            str(key): _protein_vector_to_flex(value, values_are_s2=values_are_s2)
            for key, value in source.items()
        }

    if suffix == ".npz":
        obj = np.load(path, allow_pickle=True)
        if "protein_flex" in obj and isinstance(obj["protein_flex"].item(), dict):
            source = obj["protein_flex"].item()
        else:
            source = {key: obj[key] for key in obj.files}
        return {
            str(key): _protein_vector_to_flex(value, values_are_s2=values_are_s2)
            for key, value in source.items()
        }

    raise ValueError(f"Unsupported protein flexibility file type: {path.suffix}")

