import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel
from utils.utils import load_config_file, mkdir


def _feature_frame(dataset, split, dataset_path, csv_paths=None):
    """Load only sequence identities for a cross-domain cache, or legacy splits."""
    if csv_paths is not None:
        frames = [pd.read_csv(path, usecols=['pr_id', 'Protein']) for path in csv_paths]
        return pd.concat(frames, ignore_index=True)
    if split == 'cluster':
        frames = [pd.read_csv(f'{dataset_path}{name}_with_id.csv') for name in
                  ['source_train', 'target_train', 'target_test']]
        return pd.concat(frames)
    if split in ['random', 'cold', 'augmented']:
        frames = [pd.read_csv(f'{dataset_path}{name}_with_id.csv') for name in ['train', 'val', 'test']]
        return pd.concat(frames)
    return pd.read_csv(f'./datasets/{dataset}/{dataset}_with_id.csv')


def _feature_dataset_path(dataset, split, output_dir=None):
    if output_dir is not None:
        dataset_path = str(output_dir) + '/'
        mkdir(dataset_path)
        return dataset_path
    if split in ['split_double_cold', 'split_drug_cold', 'split_protein_cold', 'split_random']:
        dataset_path = f'datasets/{dataset}/'
        mkdir(dataset_path)
        return dataset_path
    dataset_path = f'datasets/{dataset}/{split}/'
    mkdir(dataset_path)
    return dataset_path


def generate_esm2_feature(config, dataset, split, csv_paths=None, output_dir=None):
    print('start generating esm2 feature')
    dataset_path = _feature_dataset_path(dataset, split, output_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df = _feature_frame(dataset, split, dataset_path, csv_paths)

    ems2_model_path = 'models/protein/esm2_model'
    tokenizer = AutoTokenizer.from_pretrained(ems2_model_path)
    model = EsmModel.from_pretrained(ems2_model_path).to(device)
    model.eval()
    pr_features = dict()

    for protein_id in tqdm(df['pr_id'].unique(), desc='Processing'):
        protein_seq = df[df['pr_id'] == protein_id]['Protein'].iloc[0]
        inputs = tokenizer(protein_seq, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        pr_features[str(protein_id)] = outputs.last_hidden_state.squeeze()

    save_path = os.path.join(dataset_path, config.TRAIN.PR_PATH)
    with open(save_path, 'wb') as handle:
        pickle.dump(pr_features, handle)
    print('finish generating esm2 feature')


def generate_esmc_600m_feature(config, dataset, split, spec, csv_paths=None, output_dir=None):
    """Generate a separate offline ESMC-600M cache for one dataset."""
    from utils.protein_feature_backend import generate_esmc_600m_feature_cache

    dataset_path = _feature_dataset_path(dataset, split, output_dir)
    df = _feature_frame(dataset, split, dataset_path, csv_paths)
    sequences = (
        df[['pr_id', 'Protein']]
        .drop_duplicates(subset=['pr_id'])
        .set_index('pr_id')['Protein']
        .astype(str)
        .to_dict()
    )
    print(f'start generating ESMC-600M features for {len(sequences)} proteins')
    cache_path = generate_esmc_600m_feature_cache(
        sequences, spec, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print(f'finish generating ESMC-600M features: {cache_path}')
    return cache_path
