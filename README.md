# FSPR-DTI

FSPR-DTI is a sequence-based framework for drug–target interaction prediction with **flexibility-guided residue modulation** and **training-time apo–holo structural supervision**. Structural information is used only during training; inference requires only the drug SMILES, protein sequence, and sequence-derived flexibility features.

## Setup

Create a Python environment with a CUDA-compatible PyTorch installation and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Download resources

Download the datasets and fixed splits, precomputed ESM2 features, flexibility priors, processed structural supervision data, and 30 trained Full-model checkpoints from [Baidu Netdisk](https://pan.baidu.com/s/1LiXv5SjGoXmgJo-Eq22b1g) (extraction code: **2c7p**).

Extract the required ZIP files directly into the repository root, preserving their internal paths. The three `esm2_features_*.zip` archives are separated by dataset; original ESM2 pretrained weights are not required when using these cached features. See the download's README for package contents.

The expected directory structure, preprocessing requirements, and dataset boundaries are described in the [data and reproducibility guide](dataset/data_and_reproducibility.md).

## Training

Each experimental setting is defined by a complete YAML configuration file. From the project root, run:

```bash
python main.py --config configs/biosnap_cross_domain.yaml
python main.py --config configs/bindingdb_temporal_balanced.yaml
```

The configuration files specify dataset paths, model components, structural supervision settings, optimization parameters, and stopping criteria. Update the paths according to your local environment before training.

To repeat an experiment with a different random seed, copy the corresponding YAML file and modify:

- `training.TRAIN.SEED`
- `training.TRAIN.OUTPUT_DIR`
- `run.cv-results-file`

Then run the modified configuration file in the same way. The experiments reported in the paper use five random seeds, from 6 to 10.

Model checkpoints are selected according to the sum of validation AUROC and AUPRC. The test set is used only for final evaluation and is not involved in model selection.

