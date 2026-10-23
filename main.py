import argparse
import shlex
import sys
import os
import pickle
import warnings
from pathlib import Path
from time import time
import pandas as pd
import torch
from omegaconf import OmegaConf
from release_config import load_release_config
from dataloader.dataloader import DTIDataset, get_dataLoader
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from conformation_pretraining.dataset import (
    ConformationDataset, infer_gearnet_embedding_dim, make_conformation_collate,
)
from conformation_pretraining.protein_pair_batch_sampler import ProteinPairBatchSampler

from models.tapb import TAPB
from trainer_our import Trainer
from utils.utils import set_seed, mkdir, load_config_file
from utils.cv_results import numeric_metric_keys, resolve_cv_results_path
from utils.flexibility import load_drug_flex_map, load_protein_flex_map
from utils.conformation_transfer import (
    build_dti_optimizer,
    load_conformation_pretrained_modules,
)
from utils.checkpoint_provenance import (
    merge_with_conformation_pretraining_provenance,
)
from preparation import generate_esm2_feature, generate_esmc_600m_feature
from utils.cross_domain_split import cross_domain_paths
from utils.temporal_regression_split import temporal_regression_paths
from utils.temporal_binary_split import temporal_binary_paths
from utils.protein_feature_backend import (
    load_validated_protein_feature_cache,
    resolve_protein_feature_spec,
    validate_structure_supervision_feature_cache,
)
import numpy as np

#sensitive=recall
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TRAIN_CONFIG_PATH = 'configs/train_config.yaml'
MODEL_CONFIG_PATH = 'configs/model_config.yaml'
parser = argparse.ArgumentParser(description="DTI prediction")
parser.add_argument(
    '--data',
    nargs='+',
    default=['biosnap'],
    metavar='DATASET',
    help='one or more datasets, e.g. --data biosnap bindingdb'
)
parser.add_argument(
    '--split',
    nargs='+',
    default=['split_double_cold'],
    metavar='SPLIT',
    help='one or more split tasks',
    choices=[
        'random', 'cold', 'cluster', 'augmented',
        'split_double_cold', 'split_drug_cold',
        'split_protein_cold', 'split_random', 'cluster_new', 'cross_domain',
        'temporal_regression', 'temporal_binary'
    ]
)
parser.add_argument('--fold', default=None, type=int,
                    help="which fold to run, None means run all 5 folds")
parser.add_argument(
    "--train-config",
    default=TRAIN_CONFIG_PATH,
    help="path to an isolated training YAML; defaults to configs/train_config.yaml",
)
parser.add_argument("--config", default=None, help="complete dataset-specific experiment YAML")
parser.add_argument("--cv-results-file", default=None, help="optional isolated output path for the CV summary")
parser.add_argument("--drug-encoder-backend", choices=["native", "pretrained_molformer"], default=None)
parser.add_argument("--pretrained-molformer-path", default=None)
parser.add_argument("--protein-encoder-backend", choices=["esm2", "esmc_600m"], default=None)
parser.add_argument("--protein-feature-path", default=None)
parser.add_argument("--generate-protein-features", action="store_true")
parser.add_argument("--temporal-regression-root", default="datasets/tdc_bindingdb_patent/standardized", help="root containing train.csv, validation.csv and test.csv for temporal affinity regression")
parser.add_argument("--temporal-binary-root", default="datasets/bindingdb_temporal_binary_core_v1/prepared", help="root containing standardized train.csv, validation.csv and test.csv for temporal binary classification")
parser.add_argument("--task-mode", default=None, choices=["binary", "regression"], help="prediction task; temporal_regression requires regression")
parser.add_argument("--cross-domain-root", default="datasets/cross_domain", help="root containing standardized <dataset> cross-domain CSV splits")

parser.add_argument("--use-drug-flex", action="store_true", help="enable drug-level flexibility prior")
parser.add_argument("--no-drug-flex", action="store_true", help="disable drug-level flexibility prior")
parser.add_argument("--drug-flex-path", default='datasets/biosnap/drug_flex_ntabs.csv', help="CSV/PKL/NPY file with drug_id or SMILES and drug_flex_value/nTABS")
parser.add_argument("--drug-flex-dim", type=int, default=1, help="drug flexibility feature dimension, default from config")
parser.add_argument("--use-protein-flex", action="store_true", help="enable protein residue-level flexibility prior pooled into predictor")
parser.add_argument("--no-protein-flex", action="store_true", help="disable protein flexibility prior")
parser.add_argument("--protein-flex-path", default='datasets/biosnap/protein_flex_dynamine.csv', help="CSV/TSV/PKL/NPY file with protein_id/pr_id and residue-level S2 or F_p values")
parser.add_argument("--protein-flex-pooling", default=None, choices=["mean_max_std"], help="protein flexibility pooling mode")
parser.add_argument("--use-protein-induced-gate", action="store_true", help="enable residue-level protein flexibility-guided gate on protein-side cross-attention output")
parser.add_argument("--no-protein-induced-gate", action="store_true", help="disable protein induced flexibility gate")
parser.add_argument("--induced-beta-init", type=float, default=None, help="initial value for learnable induced flexibility beta")
parser.add_argument("--induced-gate-hidden-dim", type=int, default=None, help="hidden dimension of scalar induced gate MLP")
parser.add_argument("--induced-modulation-type", default=None, choices=["gate", "film"], help="induced flexibility modulation type: scalar gate or FiLM modulation")
parser.add_argument("--use-drug-induced-gate", action="store_true", help="enable drug-side flexibility-guided induced gate")
parser.add_argument("--no-drug-induced-gate", action="store_true", help="disable drug-side induced flexibility gate")
parser.add_argument("--drug-induced-beta-init", type=float, default=None, help="initial beta for drug-side induced gate")
parser.add_argument("--drug-induced-gate-hidden-dim", type=int, default=None, help="hidden dimension of drug-side scalar gate MLP")
parser.add_argument("--use-flex-attention-bias", action="store_true", help="add flexibility-conditioned bias to bidirectional cross-attention logits")
parser.add_argument("--no-flex-attention-bias", action="store_true", help="disable flexibility-conditioned attention bias")
parser.add_argument("--flex-attention-bias-hidden-dim", type=int, default=None, help="hidden dimension of flexibility attention-bias MLP")
parser.add_argument("--flex-attention-bias-init", type=float, default=None, help="initial value for learnable flexibility attention-bias scale")
parser.add_argument("--flex-attention-bias-side", default=None, choices=["both", "protein", "drug"], help="which cross-attention direction uses flexibility bias: protein means Q=H_p,K=H_d only")
parser.add_argument("--use-flex-calibration", action="store_true", help="enable protein-side pair-specific flexibility calibration after protein-side cross-attention/decoder")
parser.add_argument("--no-flex-calibration", action="store_true", help="disable pair-specific flexibility calibration")
parser.add_argument("--flex-calibration-side", default=None, choices=["protein"], help="which side to calibrate; minimal implementation supports protein only")
parser.add_argument("--flex-hidden-dim", type=int, default=None, help="hidden dimension for PairSpecificFlexibilityCalibration MLPs")
parser.add_argument("--flex-dropout", type=float, default=None, help="dropout used inside PairSpecificFlexibilityCalibration MLPs")
parser.add_argument("--flex-lambda", type=float, default=None, help="lambda_flex scale for calibration strength")
parser.add_argument("--return-flex-stats", action="store_true", help="return calibration maps/scores in model output for analysis")
parser.add_argument("--no-return-flex-stats", action="store_true", help="do not return calibration maps/scores")
parser.add_argument("--use-drug-flex-condition", action="store_true", help="condition calibration on molecule-level drug flexibility F_d")
parser.add_argument("--no-drug-flex-condition", action="store_true", help="run calibration with zero drug-flex condition instead of requiring F_d")
parser.add_argument("--protein-flex-perturbation", default=None, choices=["none", "shuffle", "random", "residue_shuffle"], help="calibration-only protein flexibility ablation; does not overwrite precomputed files")
parser.add_argument("--use-flex-film", action="store_true", help="enable protein-side flexibility-conditioned FiLM calibration of candidate cross-attention shift")
parser.add_argument("--no-flex-film", action="store_true", help="disable flexibility-conditioned FiLM calibration")
parser.add_argument("--film-side", default=None, choices=["protein", "drug", "both"], help="which side uses the new FiLM module")
parser.add_argument("--film-hidden-dim", type=int, default=None, help="hidden dimension for flexibility-conditioned FiLM")
parser.add_argument("--film-dropout", type=float, default=None, help="dropout for flexibility-conditioned FiLM")
parser.add_argument("--film-gamma-scale", type=float, default=None, help="scale applied to tanh(gamma_raw)")
parser.add_argument("--film-beta-scale", type=float, default=None, help="scale applied to tanh(beta_raw)")
parser.add_argument("--film-use-protein-flex", action="store_true", help="condition new FiLM on residue-level protein flexibility F_p")
parser.add_argument("--no-film-use-protein-flex", action="store_true", help="zero out protein flexibility input for new FiLM")
parser.add_argument("--film-use-drug-flex", action="store_true", help="condition new FiLM on molecule-level drug flexibility F_d")
parser.add_argument("--no-film-use-drug-flex", action="store_true", help="zero out drug flexibility input for new FiLM")
parser.add_argument("--protein-film-use-protein-flex", dest="protein_film_use_protein_flex", action="store_true", default=None, help="condition protein-side FiLM on residue-level protein flexibility")
parser.add_argument("--no-protein-film-use-protein-flex", dest="protein_film_use_protein_flex", action="store_false", help="disable protein flexibility input to protein-side FiLM")
parser.add_argument("--protein-film-use-drug-flex", dest="protein_film_use_drug_flex", action="store_true", default=None, help="condition protein-side FiLM on drug flexibility")
parser.add_argument("--no-protein-film-use-drug-flex", dest="protein_film_use_drug_flex", action="store_false", help="disable drug flexibility input to protein-side FiLM")
parser.add_argument("--drug-film-use-drug-flex", dest="drug_film_use_drug_flex", action="store_true", default=None, help="condition drug-side FiLM on drug flexibility")
parser.add_argument("--no-drug-film-use-drug-flex", dest="drug_film_use_drug_flex", action="store_false", help="disable drug flexibility input to drug-side FiLM")
parser.add_argument("--drug-film-use-protein-flex", dest="drug_film_use_protein_flex", action="store_true", default=None, help="condition drug-side FiLM on pooled protein flexibility")
parser.add_argument("--no-drug-film-use-protein-flex", dest="drug_film_use_protein_flex", action="store_false", help="disable protein flexibility input to drug-side FiLM")
parser.add_argument("--film-mode", default=None, choices=["scale_shift", "scale_only", "shift_only"], help="new FiLM update mode")
parser.add_argument("--film-flex-encoding", default=None, choices=["raw", "mlp"], help="how new FiLM represents F_p/F_d: raw scalar concat or MLP-encoded flexibility embeddings")
parser.add_argument("--film-flex-embed-dim", type=int, default=None, help="embedding dimension for MLP-encoded protein/drug flexibility in new FiLM")
parser.add_argument("--return-film-stats", action="store_true", help="return new FiLM gamma/beta/correction/adaptation scores in model output")
parser.add_argument("--no-return-film-stats", action="store_true", help="disable new FiLM auxiliary stats")
parser.add_argument("--film-protein-flex-perturbation", default=None, choices=["none", "shuffle", "random", "residue_shuffle"], help="FiLM-only protein flexibility ablation; does not overwrite precomputed files")
parser.add_argument("--predictor-include-induced-delta-pools", action="store_true", help="append Pool(Delta_H_d_p/H_p_d) to predictor for enabled gates")
parser.add_argument("--predictor-pooling", default=None, choices=["mean", "protein_attention"], help="DTI readout pooling: legacy masked mean or learned protein attention pooling")
parser.add_argument("--predictor-head", default=None, choices=["linear", "interaction_mlp"], help="DTI classifier: legacy linear head or nonlinear pair interaction MLP")
parser.add_argument("--predictor-dropout", type=float, default=None, help="dropout used only by predictor_head=interaction_mlp")
parser.add_argument("--compute-if-score", action="store_true", help="compute and save induced flexibility interpretability scores during eval/test")
parser.add_argument("--no-compute-if-score", action="store_true", help="disable induced flexibility score export")
parser.add_argument("--relation-alignment-enabled", action="store_true", help="enable bidirectional latent response relation alignment loss")
parser.add_argument("--no-relation-alignment", action="store_true", help="disable latent response relation alignment loss")
parser.add_argument("--relation-alignment-weight", type=float, default=0.05, help="lambda_rel for relation alignment loss")
parser.add_argument("--relation-alignment-side", default=None, choices=["both", "drug", "protein"], help="which response branch participates in relation alignment loss")
parser.add_argument("--teacher-temperature", type=float, default=None, help="teacher neighborhood softmax temperature")
parser.add_argument("--student-temperature", type=float, default=None, help="student response softmax temperature")
parser.add_argument("--response-dim", type=int, default=None, help="dimension of projected response embedding")
parser.add_argument("--alignment-target", default=None, choices=["none", "pre", "post", "response"], help="representation used for relation alignment")
parser.add_argument("--positive-only-alignment", action="store_true", help="compute relation alignment only among positive DTI samples")
parser.add_argument("--no-positive-only-alignment", action="store_true", help="compute relation alignment among all samples")
parser.add_argument("--min-positive-for-alignment", type=int, default=None, help="minimum positive samples required in a batch for relation alignment")
parser.add_argument("--save-if-score-path", default=None, help="CSV path or directory for induced flexibility score export")
parser.add_argument("--prediction-export-path", default=None, help="CSV path for checkpoint-only per-test-sample probabilities")
parser.add_argument("--target-protein-for-case-study", default=None, help="only export induced flexibility scores for this protein_id")
parser.add_argument("--if-score-alpha", type=float, default=None, help="alpha_score for S_IF = alpha*S_p_IF + (1-alpha)*S_d_IF")
parser.add_argument("--conformation-pretrained-checkpoint", default=None, help="Apo2Mol conformation-pretraining best.pt to transfer into TAPB")
parser.add_argument("--conformation-pretrained-lr-scale", type=float, default=0.1, help="learning-rate multiplier for the four transferred modules")
parser.add_argument("--allow-relation-alignment-with-conformation-pretraining", action="store_true", help="supplementary-only: retain relation alignment when loading conformation-pretrained weights")
parser.add_argument("--structure-supervision-root", default=None, help="directory containing leakage-filtered structural train.csv")
parser.add_argument("--structure-supervision-batch-size", type=int, default=4, help="structural supervision batch size")
parser.add_argument("--structure-supervision-weight", type=float, default=0.0, help="global weight for active Apo-holo structural calibration losses")
parser.add_argument("--structure-supervision-geometry-weight", type=float, default=1.0, help="inner weight for the pocket C-alpha response-magnitude term")
parser.add_argument("--structure-supervision-gearnet-root", default=None, help="directory of frozen GearNet Apo-holo embedding NPZ files; used as target-only structural supervision")
parser.add_argument("--structure-supervision-gearnet-weight", type=float, default=0.0, help="inner weight for optional frozen-GearNet response-strength supervision")
parser.add_argument("--structure-supervision-gearnet-semantic-weight", type=float, default=0.0, help="inner weight for optional residue-level frozen-GearNet change-direction supervision")
parser.add_argument("--structure-supervision-gearnet-global-weight", type=float, default=0.0, help="inner weight for pooled full-protein frozen-GearNet Apo/Holo response supervision")
parser.add_argument("--structure-supervision-gearnet-relational-weight", type=float, default=0.0, help="inner weight for same-site contrastive frozen-GearNet residue-relation supervision")
parser.add_argument("--structure-supervision-gearnet-relational-temperature", type=float, default=0.1, help="InfoNCE temperature for the relational GearNet teacher; must be positive")
parser.add_argument("--structure-supervision-gearnet-relational-candidates", type=int, default=2, help="distinct same-site drugs per relational GearNet InfoNCE candidate set; must be at least 2")
parser.add_argument("--structure-supervision-gearnet-direct-weight", type=float, default=0.0, help="inner weight for response-bound residue-level GearNet MSE-plus-cosine distillation")
parser.add_argument("--structure-supervision-gearnet-pair-weight", type=float, default=0.0, help="inner weight for teacher-weighted pocket-level GearNet MSE-plus-cosine distillation")
parser.add_argument("--structure-supervision-gearnet-pair-temperature", type=float, default=0.5, help="positive temperature for detached teacher pocket-pooling weights")
parser.add_argument("--structure-supervision-loss-normalization", choices=["none", "ema"], default="none", help="normalize active geometry and GearNet-semantic structural losses by detached EMA scales")
parser.add_argument("--structure-supervision-loss-ema-decay", type=float, default=0.99, help="EMA decay for --structure-supervision-loss-normalization ema; must be in [0, 1)")
parser.add_argument("--structure-supervision-mode", default="scalar_rmsd", choices=["scalar_rmsd", "pocket_residue", "gearnet_only", "full_protein_residue", "pocket_rms"], help="structural response supervision target and granularity")
parser.add_argument("--structure-supervision-label-perturbation", default="none", choices=["none", "batch_shuffle", "global_random_derangement"], help="target-only structural-label control; batch_shuffle rotates targets within a batch, while global_random_derangement fixes one random same-protein/site target mapping for the full run")
parser.add_argument("--structure-supervision-interval", type=int, default=1, help="run one structural update every N DTI batches")
parser.add_argument("--structure-supervision-margin", type=float, default=0.1, help="same-protein structural ranking margin")
parser.add_argument("--structure-supervision-ranking-weight", type=float, default=1.0, help="weight of same-protein ranking term inside scalar structural loss")
parser.add_argument("--structure-supervision-site-ranking-weight", type=float, default=0.0, help="weight of same-protein/same-pocket multi-ligand ranking inside pocket-residue supervision")
parser.add_argument("--structure-supervision-site-ranking-min-gap", type=float, default=0.1, help="minimum pocket C-alpha RMSD gap in Angstrom for a site-ranking pair")
parser.add_argument("--structure-supervision-residue-ranking-weight", type=float, default=0.0, help="inner weight for valid-pocket C-alpha residue-order supervision")
parser.add_argument("--structure-supervision-residue-ranking-min-gap", type=float, default=0.10, help="minimum log C-alpha displacement separation for a residue-order pair")
parser.add_argument("--structure-supervision-residue-ranking-temperature", type=float, default=0.20, help="positive logistic temperature for pocket residue-order supervision")
parser.add_argument("--structure-supervision-contact-weight-mode", default="none", choices=["none", "contact", "residue_shuffle"], help="target-only holo-ligand distance weighting for pocket C-alpha calibration")
parser.add_argument("--structure-supervision-contact-weight-alpha", type=float, default=1.0, help="non-negative contact-weight kernel amplitude")
parser.add_argument("--structure-supervision-contact-weight-sigma", type=float, default=4.5, help="positive contact-distance kernel scale in Angstrom")
parser.add_argument("--structure-supervision-contact-weight-shuffle-seed", type=int, default=0, help="fixed within-pocket contact-weight derangement seed")
parser.add_argument("--structure-supervision-apo-root", default="datasets/apo_holo/processed", help="root for Apo-holo NPZ label paths")
parser.add_argument("--structure-supervision-contact-data-root", default="datasets/apo_holo/data_folder", help="root containing holo protein PDB and ligand SDF structural samples")
parser.add_argument("--structure-supervision-esm-cache", default="datasets/apo_holo/features/esm2_features.pkl", help="Apo-holo ESM feature cache")
parser.add_argument("--structure-supervision-dynamine", default="datasets/apo_holo/features/protein_flexibility/dynamine.csv", help="Apo-holo DynaMine table")
parser.add_argument("--structure-supervision-ntabs", default="datasets/apo_holo/features/drug_flexibility/ntabs.csv", help="Apo-holo nTABS table")
parser.add_argument("--validation-only", action="store_true", help="tuning mode: train/select on validation and never evaluate test")
parser.add_argument("--zero-epoch-eval", action="store_true", help="diagnostic only: evaluate validation and test at random initialization without training")
parser.add_argument("--evaluate-checkpoint", default=None, help="restore this saved best checkpoint and evaluate target test only; never trains or overwrites the checkpoint")
parser.add_argument("--diagnostic-eval-each-epoch", action="store_true", help="diagnostic only: log target-test metrics after every training epoch")
_env_extra_args = shlex.split(os.environ.get("TAPB_MAIN_EXTRA_ARGS", ""))
_explicit_argv = _env_extra_args + sys.argv[1:]
_config_probe = argparse.ArgumentParser(add_help=False)
_config_probe.add_argument("--config")
_config_path = _config_probe.parse_known_args(_explicit_argv)[0].config
_release_config = None
if _config_path is not None:
    _release_config, _explicit_argv = load_release_config(_config_path, parser, _explicit_argv)
args = parser.parse_args(_explicit_argv)

print(f"Running on: {device}", end="\n\n")


def resolve_feature_path(path, data_folder_root, data_folder):
    if path is None or str(path).strip() == "":
        return None
    candidates = [Path(path)]
    if not Path(path).is_absolute():
        candidates.extend([Path(data_folder_root) / path, Path(data_folder) / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]



_COLD_FEATURE_SPLITS = {
    "split_double_cold",
    "split_drug_cold",
    "split_protein_cold",
    "split_random",
    "cluster_new",
}


def protein_feature_cache_root(data_name, split_name, cross_domain_root=None):
    """Return the cache root for a legacy or standardized split family."""
    if split_name == "cross_domain":
        if cross_domain_root is None:
            raise ValueError("cross_domain_root is required for cross_domain features")
        return Path(cross_domain_root) / data_name
    data_root = Path(f"datasets/{data_name}")
    return data_root if split_name in _COLD_FEATURE_SPLITS else data_root / split_name
def get_fold_paths(dataFolder, fold):
    train_path = os.path.join(dataFolder, f'train_fold{fold}.csv')
    val_path   = os.path.join(dataFolder, f'val_fold{fold}.csv')
    test_path  = os.path.join(dataFolder, f'test_fold{fold}.csv')
    return train_path, val_path, test_path

def run_one_fold(data_name, split_name, fold, config, model_configs):
    print(
        f"\n{'='*15} Dataset: {data_name} | "
        f"Split: {split_name} | Fold: {fold} {'='*15}"
    )

    torch.cuda.empty_cache()
    # print(config)
    set_seed(seed=config.TRAIN.SEED + fold)  # 推荐：不同 fold 不同 seed
    # set_seed(seed=config.TRAIN.SEED)


    if split_name == "cross_domain":
        cross_domain = cross_domain_paths(Path(args.cross_domain_root) / data_name)
        temporal = None
        dataFolder_root = str(cross_domain.root)
        dataFolder = str(cross_domain.root)
    elif split_name == "temporal_regression":
        cross_domain = None
        temporal = temporal_regression_paths(Path(args.temporal_regression_root))
        dataFolder_root = str(temporal.root)
        dataFolder = str(temporal.root)
        if str(model_configs.get("task_mode", "binary")) != "regression":
            raise ValueError("temporal_regression requires task_mode='regression'.")
        print("Temporal protocol: historical train -> historical validation -> future test; test is never used for selection.")
    elif split_name == "temporal_binary":
        cross_domain = None
        temporal = temporal_binary_paths(Path(args.temporal_binary_root))
        dataFolder_root = str(temporal.root)
        dataFolder = str(temporal.root)
        if str(model_configs.get("task_mode", "binary")) != "binary":
            raise ValueError("temporal_binary requires task_mode='binary'.")
        print("Temporal binary protocol: historical train -> historical validation -> future test; test is never used for selection.")
    else:
        cross_domain = None
        temporal = None
        dataFolder_root = f'datasets/{data_name}'
        dataFolder = os.path.join(dataFolder_root, split_name)
    mol_path = 'models/drug/molformer'

    # ===== dataset paths =====
    # ===== dataset paths =====
    if split_name == "cross_domain":
        train_path = cross_domain.train
        val_path = cross_domain.validation
        test_path = cross_domain.test
        print("Cross-domain protocol: source_train -> source_val -> target_test; target_train is not used.")
    elif split_name in {"temporal_regression", "temporal_binary"}:
        train_path = temporal.train
        val_path = temporal.validation
        test_path = temporal.test
    elif split_name == 'cluster':
        train_path = os.path.join(dataFolder, 'source_train_with_id.csv')
        val_path   = os.path.join(dataFolder, 'target_train_with_id.csv')
        test_path  = os.path.join(dataFolder, 'target_test_with_id.csv')
    elif split_name in [
        'split_double_cold', 'split_drug_cold',
        'split_protein_cold', 'split_random','cluster_new'
    ]:
        train_path, val_path, test_path = get_fold_paths(dataFolder, fold)
    else:
        train_path = os.path.join(dataFolder, 'train_with_id.csv')
        val_path   = os.path.join(dataFolder, 'val_with_id.csv')
        test_path  = os.path.join(dataFolder, 'test_with_id.csv')

    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)
    df_test = None if args.validation_only else pd.read_csv(test_path)

    # ===== protein feature backend/cache =====
    feature_root = temporal.root if temporal is not None else protein_feature_cache_root(
        data_name, split_name, args.cross_domain_root
    )
    feature_csvs = (
        cross_domain.feature_csvs if cross_domain is not None
        else temporal.feature_csvs if temporal is not None else None
    )
    protein_spec = resolve_protein_feature_spec(
        config,
        model_configs,
        feature_root,
        backend_override=args.protein_encoder_backend,
        feature_path_override=args.protein_feature_path,
    )
    model_configs["protein_encoder_backend"] = protein_spec.backend
    model_configs["protein_feature_path"] = str(protein_spec.cache_path.resolve())
    model_configs["protein_feature_dim"] = protein_spec.hidden_dim
    config.protein_encoder_backend = protein_spec.backend
    config.protein_feature_path = str(protein_spec.cache_path.resolve())
    config.protein_feature_dim = protein_spec.hidden_dim

    if not protein_spec.cache_path.is_file():
        if protein_spec.backend == "esm2":
            if args.protein_feature_path is not None:
                raise FileNotFoundError(
                    f"Requested protein feature cache does not exist: {protein_spec.cache_path}"
                )
            generate_esm2_feature(
                config, data_name, split_name,
                csv_paths=feature_csvs,
                output_dir=feature_root if feature_csvs is not None else None,
            )
        else:
            generate_esmc_600m_feature(
                config, data_name, split_name, protein_spec,
                csv_paths=feature_csvs,
                output_dir=feature_root if feature_csvs is not None else None,
            )
    pr_f = load_validated_protein_feature_cache(protein_spec)
    print(
        f"Loaded {protein_spec.backend} protein features: {len(pr_f)} from "
        f"{protein_spec.cache_path}"
    )

    # ===== model =====
    needs_gearnet_projection = (
        args.structure_supervision_gearnet_semantic_weight > 0.0
        or args.structure_supervision_gearnet_global_weight > 0.0
        or args.structure_supervision_gearnet_direct_weight > 0.0
        or args.structure_supervision_gearnet_pair_weight > 0.0
    )
    gearnet_dim = infer_gearnet_embedding_dim(Path(args.structure_supervision_gearnet_root)) if needs_gearnet_projection else 0
    model_configs["structure_gearnet_semantic_dim"] = gearnet_dim if (
        args.structure_supervision_gearnet_semantic_weight > 0.0
        or args.structure_supervision_gearnet_global_weight > 0.0
    ) else 0
    model_configs["structure_gearnet_response_distillation_dim"] = gearnet_dim if (
        args.structure_supervision_gearnet_direct_weight > 0.0 or args.structure_supervision_gearnet_pair_weight > 0.0
    ) else 0
    model = TAPB(model_configs=model_configs).to(device)
    if args.conformation_pretrained_checkpoint:
        transfer_report = load_conformation_pretrained_modules(
            model, args.conformation_pretrained_checkpoint
        )
        opt = build_dti_optimizer(
            model,
            base_lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            pretrained_lr_scale=args.conformation_pretrained_lr_scale,
            pretrained_parameter_names=transfer_report.parameter_names,
        )
        print(
            "DTI optimizer groups: "
            + ", ".join(
                f"{group.get('group_name', 'unnamed')}@lr={group['lr']:.3g}"
                for group in opt.param_groups
            )
        )
    else:
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )

    # ===== datasets =====
    drug_flex_map = None
    needs_drug_flex = (
        bool(model_configs.get('use_drug_flex', False))
        or bool(model_configs.get('use_drug_induced_gate', False))
        or bool(model_configs.get('use_flex_attention_bias', False))
        or (
            bool(model_configs.get('use_flex_calibration', False))
            and bool(model_configs.get('use_drug_flex_condition', True))
        )
        or (
            bool(model_configs.get('use_flex_film', False))
            and (
                (
                    model_configs.get('film_side', 'protein') in {'protein', 'both'}
                    and bool(model_configs.get(
                        'protein_film_use_drug_flex',
                        model_configs.get('film_use_drug_flex', True),
                    ))
                )
                or (
                    model_configs.get('film_side', 'protein') in {'drug', 'both'}
                    and bool(model_configs.get(
                        'drug_film_use_drug_flex',
                        model_configs.get('film_use_drug_flex', True),
                    ))
                )
            )
        )
        or bool(model_configs.get('compute_if_score', False))
    )
    if needs_drug_flex:
        configured_flex_path = model_configs.get('drug_flex_path', '')
        if not configured_flex_path:
            configured_flex_path = 'drug_flex_ntabs.csv'
        drug_flex_path = resolve_feature_path(
            configured_flex_path,
            dataFolder_root,
            dataFolder,
        )
        if drug_flex_path is None:
            raise ValueError('use_drug_flex=True but drug_flex_path is empty.')
        drug_flex_map = load_drug_flex_map(
            drug_flex_path,
            flex_dim=int(model_configs.get('drug_flex_dim', 1)),
            log1p=bool(model_configs.get('drug_flex_log1p', True)),
        )
        print(f"Loaded drug flexibility priors: {len(drug_flex_map)} from {drug_flex_path}")

    protein_flex_map = None
    needs_protein_flex = (
        bool(model_configs.get('use_protein_flex', False))
        or bool(model_configs.get('use_protein_induced_gate', False))
        or bool(model_configs.get('use_drug_induced_gate', False))
        or bool(model_configs.get('use_flex_attention_bias', False))
        or bool(model_configs.get('use_flex_calibration', False))
        or (
            bool(model_configs.get('use_flex_film', False))
            and (
                (
                    model_configs.get('film_side', 'protein') in {'protein', 'both'}
                    and bool(model_configs.get(
                        'protein_film_use_protein_flex',
                        model_configs.get('film_use_protein_flex', True),
                    ))
                )
                or (
                    model_configs.get('film_side', 'protein') in {'drug', 'both'}
                    and bool(model_configs.get(
                        'drug_film_use_protein_flex',
                        model_configs.get('film_use_protein_flex', True),
                    ))
                )
            )
        )
        or bool(model_configs.get('compute_if_score', False))
    )
    if needs_protein_flex:
        configured_protein_flex_path = model_configs.get('protein_flex_path', '')
        if not configured_protein_flex_path:
            raise ValueError(
                'Protein flexibility is required but protein_flex_path is empty. '
                'Please pass --protein-flex-path or set protein_flex_path in configs/model_config.yaml.'
            )
        protein_flex_path = resolve_feature_path(
            configured_protein_flex_path,
            dataFolder_root,
            dataFolder,
        )
        protein_flex_map = load_protein_flex_map(protein_flex_path, values_are_s2=True)
        print(f"Loaded protein flexibility priors: {len(protein_flex_map)} from {protein_flex_path}")

    train_dataset = DTIDataset(
        df_train.index.values, df_train, pr_f,
        drug_flex_map=drug_flex_map,
        protein_flex_map=protein_flex_map,
    )
    val_dataset   = DTIDataset(
        df_val.index.values, df_val, pr_f,
        drug_flex_map=drug_flex_map,
        protein_flex_map=protein_flex_map,
    )
    test_dataset = None if args.validation_only else DTIDataset(
        df_test.index.values, df_test, pr_f,
        drug_flex_map=drug_flex_map,
        protein_flex_map=protein_flex_map,
    )

    drug_tokenizer = AutoTokenizer.from_pretrained(
        mol_path, trust_remote_code=True
    )

    structural_loader = None
    if args.structure_supervision_root is not None and not args.evaluate_checkpoint:
        structural_root = Path(args.structure_supervision_root)
        structural_train_csv = structural_root / "train.csv"
        if not structural_train_csv.is_file():
            raise FileNotFoundError(
                f"Structural supervision train.csv was not found: {structural_train_csv}"
            )
        structural_dataset = ConformationDataset(
            split_csv=structural_train_csv,
            apo_root=Path(args.structure_supervision_apo_root),
            esm_cache=Path(args.structure_supervision_esm_cache),
            dynamine_path=Path(args.structure_supervision_dynamine),
            ntabs_path=Path(args.structure_supervision_ntabs),
            protein_feature_dim=protein_spec.hidden_dim,
            gearnet_root=(None if args.structure_supervision_gearnet_root is None else Path(args.structure_supervision_gearnet_root)),
            contact_data_root=(
                Path(args.structure_supervision_contact_data_root)
                if args.structure_supervision_contact_weight_mode != "none" else None
            ),
            structural_target_permutation=(
                "global_random_derangement"
                if args.structure_supervision_label_perturbation == "global_random_derangement"
                else "none"
            ),
            structural_target_permutation_seed=int(config.TRAIN.SEED) + fold,
        )
        structural_loader = DataLoader(
            structural_dataset,
            batch_sampler=ProteinPairBatchSampler(
                structural_dataset.rows["protein_id"].astype(str).tolist(),
                batch_size=args.structure_supervision_batch_size,
                seed=int(config.TRAIN.SEED) + fold,
                group_ids=(
                    structural_dataset.site_group_ids
                    if (float(args.structure_supervision_site_ranking_weight) > 0.0
                        or float(args.structure_supervision_gearnet_relational_weight) > 0.0
                        or args.structure_supervision_label_perturbation == "global_random_derangement")
                    else None
                ),
                pair_drug_ids=(
                    structural_dataset.rows["drug_id"].astype(str).tolist()
                    if float(args.structure_supervision_gearnet_relational_weight) > 0.0
                    else None
                ),
                drop_unpaired=float(args.structure_supervision_gearnet_relational_weight) > 0.0,
                max_distinct_drugs_per_batch=(
                    args.structure_supervision_gearnet_relational_candidates
                    if float(args.structure_supervision_gearnet_relational_weight) > 0.0 else 2
                ),
            ),
            num_workers=0,
            collate_fn=make_conformation_collate(drug_tokenizer),
        )
        print(
            f"Loaded leakage-filtered structural supervision: "
            f"{len(structural_dataset)} rows from {structural_train_csv}"
        )

    bz  = config.TRAIN.BATCH_SIZE
    train_loader = get_dataLoader(
        bz,
        train_dataset,
        drug_tokenizer,
        shuffle=True,
        MLM=bool(getattr(config.TRAIN, 'MLM', False)),
        mask_rate=float(getattr(config.TRAIN, 'MASK_PROBABILITY', 0.0)),
        target_random_deletion_ratio=float(
            getattr(config.TRAIN, 'TARGET_RANDOM_DROP_RATIO', 0.0)
        ),
    )
    val_loader  = get_dataLoader(bz, val_dataset, drug_tokenizer)
    test_loader = None if args.validation_only else get_dataLoader(
        bz, test_dataset, drug_tokenizer
    )

    # ===== output dir (per fold) =====
    output_path = (
        f"./results/{data_name}/{split_name}/"
        f"fold{fold}/{config.TRAIN.OUTPUT_DIR}{config.TRAIN.SEED}"
    )
    mkdir(output_path)

    trainer = Trainer(
        model, opt, device,
        train_loader, val_loader, test_loader,
        output_path, config,
        structural_dataloader=structural_loader,
        structural_weight=args.structure_supervision_weight,
        structural_geometry_weight=args.structure_supervision_geometry_weight,
        structural_interval=args.structure_supervision_interval,
        structural_margin=args.structure_supervision_margin,
        structural_ranking_weight=args.structure_supervision_ranking_weight,
        structural_site_ranking_weight=args.structure_supervision_site_ranking_weight,
        structural_site_ranking_min_gap=args.structure_supervision_site_ranking_min_gap,
        structural_residue_ranking_weight=args.structure_supervision_residue_ranking_weight,
        structural_residue_ranking_min_gap=args.structure_supervision_residue_ranking_min_gap,
        structural_residue_ranking_temperature=args.structure_supervision_residue_ranking_temperature,
        structural_label_perturbation=args.structure_supervision_label_perturbation,
        structural_mode=args.structure_supervision_mode,
        structural_gearnet_weight=args.structure_supervision_gearnet_weight,
        structural_gearnet_semantic_weight=args.structure_supervision_gearnet_semantic_weight,
        structural_gearnet_global_weight=args.structure_supervision_gearnet_global_weight,
        structural_gearnet_relational_weight=args.structure_supervision_gearnet_relational_weight,
        structural_gearnet_relational_temperature=args.structure_supervision_gearnet_relational_temperature,
        structural_gearnet_direct_weight=args.structure_supervision_gearnet_direct_weight,
        structural_gearnet_pair_weight=args.structure_supervision_gearnet_pair_weight,
        structural_gearnet_pair_temperature=args.structure_supervision_gearnet_pair_temperature,
        structural_loss_normalization=args.structure_supervision_loss_normalization,
        structural_loss_ema_decay=args.structure_supervision_loss_ema_decay,
        structural_contact_weight_mode=args.structure_supervision_contact_weight_mode,
        structural_contact_weight_alpha=args.structure_supervision_contact_weight_alpha,
        structural_contact_weight_sigma=args.structure_supervision_contact_weight_sigma,
        structural_contact_weight_shuffle_seed=args.structure_supervision_contact_weight_shuffle_seed,
        diagnostic_eval_each_epoch=args.diagnostic_eval_each_epoch,
        zero_epoch_eval=args.zero_epoch_eval,
        validation_only=args.validation_only,
        prediction_export_path=args.prediction_export_path,
    )

    if args.evaluate_checkpoint:
        result, best_epoch = trainer.evaluate_checkpoint(args.evaluate_checkpoint)
    else:
        result, best_epoch = trainer.train()

    return result


def run_experiment(data_name, split_name, config, model_configs):
    all_results = []
    if args.fold is not None:
        folds = [args.fold]
    elif split_name in {"temporal_regression", "temporal_binary"}:
        folds = [1]
    else:
        folds = list(range(1, 6))

    for fold in folds:
        result = run_one_fold(
            data_name, split_name, fold, config, model_configs
        )
        all_results.append(result)

    print("\n" + "=" * 60)
    print(f"Results | Dataset: {data_name} | Split: {split_name}")
    print("=" * 60)

    metrics = numeric_metric_keys(all_results[0])
    for metric in metrics:
        values = [result[metric] for result in all_results]
        print(
            f"{metric.upper():12s}: "
            f"{np.mean(values):.4f} ± {np.std(values):.4f}"
        )

    if args.evaluate_checkpoint:
        print("\nCheckpoint-only evaluation complete; CV summary was not written.")
        return all_results

    save_path = resolve_cv_results_path(args, config, data_name, split_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(f"Dataset: {data_name}\n")
        f.write(f"Split: {split_name}\n")
        f.write(f"Folds: {folds}\n")
        f.write("=" * 60 + "\n")
        for metric in metrics:
            values = [result[metric] for result in all_results]
            line = (
                f"{metric.upper():12s}: "
                f"{np.mean(values):.4f} ± {np.std(values):.4f}\n"
            )
            f.write(line)

    print(f"\n[Saved] CV results saved to: {save_path}")
    return all_results


# =========================
# Main (multi-dataset / multi-split CV controller)
# =========================
def main():
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    if args.fold is not None:
        assert 1 <= args.fold <= 5, "--fold must be between 1 and 5"
    if args.evaluate_checkpoint and args.validation_only:
        raise ValueError("--evaluate-checkpoint cannot be combined with --validation-only")
    if args.evaluate_checkpoint and args.zero_epoch_eval:
        raise ValueError("--evaluate-checkpoint cannot be combined with --zero-epoch-eval")
    if args.structure_supervision_root is None and args.structure_supervision_weight != 0.0:
        raise ValueError("--structure-supervision-weight requires --structure-supervision-root")
    if args.structure_supervision_root is not None and args.structure_supervision_weight <= 0.0:
        raise ValueError("--structure-supervision-root requires a positive --structure-supervision-weight")
    if args.structure_supervision_geometry_weight < 0.0:
        raise ValueError("--structure-supervision-geometry-weight must be non-negative")
    if (args.structure_supervision_root is not None
            and args.structure_supervision_geometry_weight <= 0.0
            and args.structure_supervision_gearnet_weight <= 0.0
            and args.structure_supervision_gearnet_semantic_weight <= 0.0
            and args.structure_supervision_gearnet_global_weight <= 0.0
            and args.structure_supervision_gearnet_relational_weight <= 0.0
            and args.structure_supervision_gearnet_direct_weight <= 0.0
            and args.structure_supervision_gearnet_pair_weight <= 0.0):
        raise ValueError("--structure-supervision-root requires an active geometry or GearNet structural term")
    if args.structure_supervision_batch_size < 1:
        raise ValueError("--structure-supervision-batch-size must be positive")
    if args.structure_supervision_contact_weight_alpha < 0.0:
        raise ValueError("--structure-supervision-contact-weight-alpha must be non-negative")
    if args.structure_supervision_contact_weight_sigma <= 0.0:
        raise ValueError("--structure-supervision-contact-weight-sigma must be positive")
    if (args.structure_supervision_contact_weight_mode != "none"
            and args.structure_supervision_mode != "pocket_residue"):
        raise ValueError("contact weighting requires --structure-supervision-mode pocket_residue")
    if args.structure_supervision_interval < 1:
        raise ValueError("--structure-supervision-interval must be positive")
    if args.structure_supervision_gearnet_weight < 0.0:
        raise ValueError("--structure-supervision-gearnet-weight must be non-negative")
    if args.structure_supervision_gearnet_semantic_weight < 0.0:
        raise ValueError("--structure-supervision-gearnet-semantic-weight must be non-negative")
    if args.structure_supervision_gearnet_global_weight < 0.0:
        raise ValueError("--structure-supervision-gearnet-global-weight must be non-negative")
    if args.structure_supervision_gearnet_relational_weight < 0.0:
        raise ValueError("--structure-supervision-gearnet-relational-weight must be non-negative")
    if args.structure_supervision_gearnet_relational_temperature <= 0.0:
        raise ValueError("--structure-supervision-gearnet-relational-temperature must be positive")
    if args.structure_supervision_gearnet_direct_weight < 0.0:
        raise ValueError("--structure-supervision-gearnet-direct-weight must be non-negative")
    if args.structure_supervision_gearnet_pair_weight < 0.0:
        raise ValueError("--structure-supervision-gearnet-pair-weight must be non-negative")
    if args.structure_supervision_gearnet_pair_temperature <= 0.0:
        raise ValueError("--structure-supervision-gearnet-pair-temperature must be positive")
    if args.structure_supervision_gearnet_relational_candidates < 2:
        raise ValueError("--structure-supervision-gearnet-relational-candidates must be at least 2")
    if args.structure_supervision_gearnet_relational_candidates > args.structure_supervision_batch_size:
        raise ValueError("--structure-supervision-gearnet-relational-candidates cannot exceed --structure-supervision-batch-size")
    if (args.structure_supervision_gearnet_semantic_weight > 0.0
            and args.structure_supervision_gearnet_relational_weight > 0.0):
        raise ValueError("direct and relational GearNet semantic weights are mutually exclusive")
    if not 0.0 <= args.structure_supervision_loss_ema_decay < 1.0:
        raise ValueError("--structure-supervision-loss-ema-decay must be in [0, 1)")
    if args.structure_supervision_gearnet_semantic_weight > 0.0:
        if args.structure_supervision_root is None:
            raise ValueError("--structure-supervision-gearnet-semantic-weight requires --structure-supervision-root")
        if args.structure_supervision_gearnet_root is None:
            raise ValueError("--structure-supervision-gearnet-semantic-weight requires --structure-supervision-gearnet-root")
        if args.structure_supervision_mode != "pocket_residue":
            raise ValueError("--structure-supervision-gearnet-semantic-weight requires pocket_residue mode")
    if args.structure_supervision_gearnet_global_weight > 0.0:
        if args.structure_supervision_root is None:
            raise ValueError("--structure-supervision-gearnet-global-weight requires --structure-supervision-root")
        if args.structure_supervision_gearnet_root is None:
            raise ValueError("--structure-supervision-gearnet-global-weight requires --structure-supervision-gearnet-root")
        if args.structure_supervision_mode != "pocket_residue":
            raise ValueError("--structure-supervision-gearnet-global-weight requires pocket_residue mode")
    if args.structure_supervision_gearnet_relational_weight > 0.0:
        if args.structure_supervision_root is None:
            raise ValueError("--structure-supervision-gearnet-relational-weight requires --structure-supervision-root")
        if args.structure_supervision_gearnet_root is None:
            raise ValueError("--structure-supervision-gearnet-relational-weight requires --structure-supervision-gearnet-root")
        if args.structure_supervision_mode != "pocket_residue":
            raise ValueError("--structure-supervision-gearnet-relational-weight requires pocket_residue mode")
    if args.structure_supervision_gearnet_direct_weight > 0.0 or args.structure_supervision_gearnet_pair_weight > 0.0:
        if args.structure_supervision_root is None:
            raise ValueError("response-bound GearNet distillation requires --structure-supervision-root")
        if args.structure_supervision_gearnet_root is None:
            raise ValueError("response-bound GearNet distillation requires --structure-supervision-gearnet-root")
        if args.structure_supervision_mode != "pocket_residue":
            raise ValueError("response-bound GearNet distillation requires pocket_residue mode")
    if args.structure_supervision_gearnet_weight > 0.0:
        if args.structure_supervision_root is None:
            raise ValueError("--structure-supervision-gearnet-weight requires --structure-supervision-root")
        if args.structure_supervision_gearnet_root is None:
            raise ValueError("--structure-supervision-gearnet-weight requires --structure-supervision-gearnet-root")
        if args.structure_supervision_mode not in {"pocket_residue", "gearnet_only"}:
            raise ValueError("--structure-supervision-gearnet-weight requires pocket_residue or gearnet_only mode")
    if args.structure_supervision_mode == "gearnet_only":
        if args.structure_supervision_gearnet_root is None:
            raise ValueError("--structure-supervision-mode gearnet_only requires --structure-supervision-gearnet-root")
        if args.structure_supervision_gearnet_weight <= 0.0:
            raise ValueError("--structure-supervision-mode gearnet_only requires a positive --structure-supervision-gearnet-weight")

    train_config = (
        _release_config.training
        if _release_config is not None
        else load_config_file(args.train_config)
    )
    model_config = load_config_file(MODEL_CONFIG_PATH)
    model_config.task_mode = str(getattr(train_config, "task_mode", "binary"))
    if args.task_mode is not None:
        model_config.task_mode = args.task_mode
    if args.drug_encoder_backend is not None:
        model_config.drug_encoder_backend = args.drug_encoder_backend
    if args.pretrained_molformer_path is not None:
        model_config.pretrained_molformer_path = args.pretrained_molformer_path
    if args.protein_encoder_backend is not None:
        model_config.protein_encoder_backend = args.protein_encoder_backend
    if args.use_drug_flex:
        model_config.use_drug_flex = True
    if args.no_drug_flex:
        model_config.use_drug_flex = False
    if args.drug_flex_path is not None:
        model_config.drug_flex_path = args.drug_flex_path
    if args.drug_flex_dim is not None:
        model_config.drug_flex_dim = args.drug_flex_dim
    if args.use_protein_flex:
        model_config.use_protein_flex = True
    if args.no_protein_flex:
        model_config.use_protein_flex = False
    if args.protein_flex_path is not None:
        model_config.protein_flex_path = args.protein_flex_path
    if args.protein_flex_pooling is not None:
        model_config.protein_flex_pooling = args.protein_flex_pooling
    if args.use_protein_induced_gate:
        model_config.use_protein_induced_gate = True
    if args.no_protein_induced_gate:
        model_config.use_protein_induced_gate = False
    if args.induced_beta_init is not None:
        model_config.induced_beta_init = args.induced_beta_init
    if args.induced_gate_hidden_dim is not None:
        model_config.induced_gate_hidden_dim = args.induced_gate_hidden_dim
    if args.induced_modulation_type is not None:
        model_config.induced_modulation_type = args.induced_modulation_type
    if args.use_drug_induced_gate:
        model_config.use_drug_induced_gate = True
    if args.no_drug_induced_gate:
        model_config.use_drug_induced_gate = False
    if args.drug_induced_beta_init is not None:
        model_config.drug_induced_beta_init = args.drug_induced_beta_init
    if args.drug_induced_gate_hidden_dim is not None:
        model_config.drug_induced_gate_hidden_dim = args.drug_induced_gate_hidden_dim
    if args.use_flex_attention_bias:
        model_config.use_flex_attention_bias = True
    if args.no_flex_attention_bias:
        model_config.use_flex_attention_bias = False
    if args.flex_attention_bias_hidden_dim is not None:
        model_config.flex_attention_bias_hidden_dim = args.flex_attention_bias_hidden_dim
    if args.flex_attention_bias_init is not None:
        model_config.flex_attention_bias_init = args.flex_attention_bias_init
    if args.flex_attention_bias_side is not None:
        model_config.flex_attention_bias_side = args.flex_attention_bias_side
    if args.use_flex_calibration:
        model_config.use_flex_calibration = True
    if args.no_flex_calibration:
        model_config.use_flex_calibration = False
    if args.flex_calibration_side is not None:
        model_config.flex_calibration_side = args.flex_calibration_side
    if args.flex_hidden_dim is not None:
        model_config.flex_hidden_dim = args.flex_hidden_dim
    if args.flex_dropout is not None:
        model_config.flex_dropout = args.flex_dropout
    if args.flex_lambda is not None:
        model_config.flex_lambda = args.flex_lambda
    if args.return_flex_stats:
        model_config.return_flex_stats = True
    if args.no_return_flex_stats:
        model_config.return_flex_stats = False
    if args.use_drug_flex_condition:
        model_config.use_drug_flex_condition = True
    if args.no_drug_flex_condition:
        model_config.use_drug_flex_condition = False
    if args.protein_flex_perturbation is not None:
        model_config.protein_flex_perturbation = args.protein_flex_perturbation
    if args.use_flex_film:
        model_config.use_flex_film = True
    if args.no_flex_film:
        model_config.use_flex_film = False
    if args.film_side is not None:
        model_config.film_side = args.film_side
    if args.film_hidden_dim is not None:
        model_config.film_hidden_dim = args.film_hidden_dim
    if args.film_dropout is not None:
        model_config.film_dropout = args.film_dropout
    if args.film_gamma_scale is not None:
        model_config.film_gamma_scale = args.film_gamma_scale
    if args.film_beta_scale is not None:
        model_config.film_beta_scale = args.film_beta_scale
    if args.film_use_protein_flex:
        model_config.film_use_protein_flex = True
    if args.no_film_use_protein_flex:
        model_config.film_use_protein_flex = False
    if args.film_use_drug_flex:
        model_config.film_use_drug_flex = True
    if args.no_film_use_drug_flex:
        model_config.film_use_drug_flex = False
    for field in (
        "protein_film_use_protein_flex",
        "protein_film_use_drug_flex",
        "drug_film_use_drug_flex",
        "drug_film_use_protein_flex",
    ):
        value = getattr(args, field)
        if value is not None:
            model_config[field] = bool(value)
    if args.film_mode is not None:
        model_config.film_mode = args.film_mode
    if args.film_flex_encoding is not None:
        model_config.film_flex_encoding = args.film_flex_encoding
    if args.film_flex_embed_dim is not None:
        model_config.film_flex_embed_dim = args.film_flex_embed_dim
    if args.return_film_stats:
        model_config.return_film_stats = True
    if args.no_return_film_stats:
        model_config.return_film_stats = False
    if args.film_protein_flex_perturbation is not None:
        model_config.film_protein_flex_perturbation = args.film_protein_flex_perturbation
    if args.predictor_include_induced_delta_pools:
        model_config.predictor_include_induced_delta_pools = True
    if args.predictor_pooling is not None:
        model_config.predictor_pooling = args.predictor_pooling
    if args.predictor_head is not None:
        model_config.predictor_head = args.predictor_head
    if args.predictor_dropout is not None:
        model_config.predictor_dropout = args.predictor_dropout
    if args.compute_if_score:
        model_config.compute_if_score = True
    if args.no_compute_if_score:
        model_config.compute_if_score = False
    if args.relation_alignment_enabled:
        model_config.relation_alignment_enabled = True
        if args.alignment_target is None:
            model_config.alignment_target = "response"
    if args.no_relation_alignment:
        model_config.relation_alignment_enabled = False
        model_config.alignment_target = "none"
    if args.relation_alignment_weight is not None:
        model_config.relation_alignment_weight = args.relation_alignment_weight
    if args.relation_alignment_side is not None:
        model_config.relation_alignment_side = args.relation_alignment_side
    if args.teacher_temperature is not None:
        model_config.teacher_temperature = args.teacher_temperature
    if args.student_temperature is not None:
        model_config.student_temperature = args.student_temperature
    if args.response_dim is not None:
        model_config.response_dim = args.response_dim
    if args.alignment_target is not None:
        model_config.alignment_target = args.alignment_target
        if args.alignment_target != "none":
            model_config.relation_alignment_enabled = True
    if args.positive_only_alignment:
        model_config.positive_only_alignment = True
    if args.no_positive_only_alignment:
        model_config.positive_only_alignment = False
    if args.min_positive_for_alignment is not None:
        model_config.min_positive_for_alignment = args.min_positive_for_alignment
    if args.save_if_score_path is not None:
        model_config.save_if_score_path = args.save_if_score_path
    if args.target_protein_for_case_study is not None:
        model_config.target_protein_for_case_study = args.target_protein_for_case_study
    if args.if_score_alpha is not None:
        model_config.if_score_alpha = args.if_score_alpha
    if args.conformation_pretrained_checkpoint:
        required = {
            "use_flex_film": bool(model_config.get("use_flex_film", False)),
            "protein_film_side": model_config.get("film_side") in {"protein", "both"},
            "shift_only": model_config.get("film_mode") == "shift_only",
            "protein_flex_condition": bool(model_config.get("film_use_protein_flex", False)),
            "drug_flex_condition": bool(model_config.get("film_use_drug_flex", False)),
        }
        failed = [name for name, enabled in required.items() if not enabled]
        if failed:
            raise ValueError(
                "Conformation-pretrained transfer requires protein-side shift_only FiLM "
                f"with protein/drug flexibility enabled; failed checks: {failed}"
            )
        if not args.allow_relation_alignment_with_conformation_pretraining:
            model_config.relation_alignment_enabled = False
            model_config.alignment_target = "none"
            print("Conformation-pretrained formal configuration: relation alignment disabled.")
    config = merge_with_conformation_pretraining_provenance(
        train_config,
        model_config,
        args.conformation_pretrained_checkpoint,
        merge_fn=OmegaConf.merge,
    )
    if args.structure_supervision_root is not None:
        validate_structure_supervision_feature_cache(
            str(model_config.get("protein_encoder_backend", "esm2")),
            Path(args.structure_supervision_esm_cache),
        )
        config.TRAIN.structure_supervision = {
            "enabled": True,
            "root": str(Path(args.structure_supervision_root).resolve()),
            "batch_size": int(args.structure_supervision_batch_size),
            "weight": float(args.structure_supervision_weight),
            "geometry_weight": float(args.structure_supervision_geometry_weight),
            "interval": int(args.structure_supervision_interval),
            "margin": float(args.structure_supervision_margin),
            "label_perturbation": str(args.structure_supervision_label_perturbation),
            "ranking_weight": float(args.structure_supervision_ranking_weight),
            "site_ranking_weight": float(args.structure_supervision_site_ranking_weight),
            "site_ranking_min_gap": float(args.structure_supervision_site_ranking_min_gap),
            "mode": str(args.structure_supervision_mode),
            "apo_root": str(Path(args.structure_supervision_apo_root).resolve()),
            "esm_cache": str(Path(args.structure_supervision_esm_cache).resolve()),
            "dynamine": str(Path(args.structure_supervision_dynamine).resolve()),
            "ntabs": str(Path(args.structure_supervision_ntabs).resolve()),
            "target_policy": "target_only_not_model_input",
        }
    else:
        config.TRAIN.structure_supervision = {"enabled": False}
    model_configs = dict(model_config)

    experiments = [
        (data_name, split_name)
        for data_name in args.data
        for split_name in args.split
    ]
    if args.generate_protein_features:
        for data_name, split_name in experiments:
            paths = (
                cross_domain_paths(Path(args.cross_domain_root) / data_name)
                if split_name == "cross_domain"
                else temporal_regression_paths(Path(args.temporal_regression_root))
                if split_name == "temporal_regression"
                else temporal_binary_paths(Path(args.temporal_binary_root))
                if split_name == "temporal_binary" else None
            )
            feature_root = paths.root if paths is not None else protein_feature_cache_root(
                data_name, split_name, args.cross_domain_root
            )
            protein_spec = resolve_protein_feature_spec(
                config, model_configs, feature_root,
                backend_override=args.protein_encoder_backend,
                feature_path_override=args.protein_feature_path,
            )
            if protein_spec.cache_path.is_file():
                load_validated_protein_feature_cache(protein_spec)
                print(f"Protein feature cache already valid: {protein_spec.cache_path}")
                continue
            if protein_spec.backend == "esm2":
                if args.protein_feature_path is not None:
                    raise FileNotFoundError(f"Requested protein feature cache does not exist: {protein_spec.cache_path}")
                generate_esm2_feature(config, data_name, split_name, csv_paths=paths.feature_csvs if paths else None, output_dir=paths.root if paths else None)
            else:
                generate_esmc_600m_feature(config, data_name, split_name, protein_spec, csv_paths=paths.feature_csvs if paths else None, output_dir=paths.root if paths else None)
        return {}

    print(f"Experiment queue ({len(experiments)} total):")
    for index, (data_name, split_name) in enumerate(experiments, start=1):
        print(f"  {index}. dataset={data_name}, split={split_name}")

    results_by_experiment = {}
    for index, (data_name, split_name) in enumerate(experiments, start=1):
        print(
            f"\n{'#'*70}\n"
            f"Experiment {index}/{len(experiments)}: "
            f"dataset={data_name}, split={split_name}\n"
            f"{'#'*70}"
        )
        results_by_experiment[(data_name, split_name)] = run_experiment(
            data_name, split_name, config, model_configs
        )

    print("\n" + "=" * 70)
    print(f"Completed all {len(experiments)} experiments.")
    print("=" * 70)
    return results_by_experiment


if __name__ == '__main__':
    s = time()
    main()
    e = time()
    print(f"\nTotal running time: {round(e - s, 2)}s")
