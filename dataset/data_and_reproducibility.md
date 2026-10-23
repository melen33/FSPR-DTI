# Data and reproducibility

Large data and model files are distributed separately from this code package through [Baidu Netdisk](https://pan.baidu.com/s/1LiXv5SjGoXmgJo-Eq22b1g) (extraction code: **2c7p**). Extract the required ZIP archives into the repository root without adding an extra enclosing directory. Preserve the supplied train/validation/test boundaries and observe the original resources' applicable terms.

## BioSNAP cross-domain

Pass `--cross-domain-root datasets/cross_domain` and place `source_train.csv`, `source_val.csv`, `target_train.csv` and `target_test.csv` under `datasets/cross_domain/biosnap/`. The source train/validation files are used for fitting and selection; target train is not used in this protocol. Provide the frozen ESM2 residue-feature cache under that split directory, a residue-level DynaMine flexibility-prior file using `--protein-flex-path`, and the leakage-filtered apo–holo structural supervision directory using `--structure-supervision-root`.

## BindingDB balanced temporal binary

Pass `--temporal-binary-root datasets/bindingdb_temporal_balanced_v1/prepared`. This directory must contain standardized `train.csv`, `validation.csv`, and `test.csv` with at least `SMILES`, `Protein`, `Y`, `drug_id` and `pr_id`. Provide the corresponding ESM2 cache with `--protein-feature-path`, the residue flexibility prior with `--protein-flex-path`, and the filtered structural training labels with `--structure-supervision-root`. Do not use future/test interaction labels to fit the model, choose parameters, or create structural training targets.

## Structural teacher and model inputs

The structural dataset is an auxiliary **training-only** teacher. Its `train.csv` and referenced apo/holo processed records must be present in the structure-supervision directory. Inference uses SMILES, sequence-derived ESM2 residue features and sequence-derived flexibility priors, not apo/holo coordinates. These generated features are still required at inference and are provided in the separate resource download.


