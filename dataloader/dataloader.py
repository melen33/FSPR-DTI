import torch
from torch.nn.utils.rnn import pad_sequence as pad
from torch.utils.data import Dataset, DataLoader
import random
import copy
from rdkit import Chem
import numpy as np
import warnings

class DTIDataset(Dataset):
    def __init__(self, list_IDs, df, pr_features, drug_flex_map=None, protein_flex_map=None, protein_flex_fill_value=0.0):
        self.list_IDs = list_IDs
        self.df = df
        self.pr_features = pr_features
        self.drug_flex_map = drug_flex_map
        self.protein_flex_map = protein_flex_map
        self.protein_flex_fill_value = float(protein_flex_fill_value)
        self._warned_missing_protein_flex = set()
        self._warned_shape_protein_flex = set()
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        index = self.list_IDs[idx]
        # smiles
        SMILES = self.df.iloc[index]['SMILES']
        # SMILES = randomize_smile(SMILES)
        # proteins
        pr_id = self.df.iloc[index]['pr_id']
        pr_seq = self.df.iloc[index]['Protein']
        Protein = self._get_protein_feature(pr_id)
        # labels
        y = self.df.iloc[index]["Y"]
        row = self.df.iloc[index]
        sample = {
            'SMILES': SMILES,
            'Protein': Protein,
            'Protein_seq': pr_seq,
            'Y': y,
            'protein_id': str(pr_id),
            'drug_id': str(row['drug_id']) if 'drug_id' in row.index else str(row['SMILES']),
        }
        if self.drug_flex_map is not None:
            sample['drug_flex'] = self._get_drug_flex(self.df.iloc[index])
        if self.protein_flex_map is not None:
            # protein_flex: [L_p, 1], aligned with the ESM protein feature tokens.
            sample['protein_flex'] = self._get_protein_flex(pr_id, Protein)
        return sample

    def _get_drug_flex(self, row):
        # Prefer explicit drug IDs when present, otherwise fall back to SMILES.
        for column in ('drug_id', 'DrugBank ID', 'DrugBank_ID', 'SMILES'):
            if column in row.index:
                key = str(row[column])
                if key in self.drug_flex_map:
                    return torch.as_tensor(self.drug_flex_map[key], dtype=torch.float32)
        raise KeyError(
            'Drug flexibility value not found. Expected one of drug_id, '
            'DrugBank ID, DrugBank_ID, or SMILES to match the drug_flex map.'
        )

    def _get_protein_flex(self, pr_id, protein_feature):
        key_candidates = [str(pr_id), pr_id]
        try:
            key_candidates.append(int(pr_id))
        except (TypeError, ValueError):
            pass

        value = None
        matched_key = None
        for key in key_candidates:
            if key in self.protein_flex_map:
                value = self.protein_flex_map[key]
                matched_key = key
                break

        expected_len = int(protein_feature.shape[0])
        if value is None:
            if str(pr_id) not in self._warned_missing_protein_flex:
                warnings.warn(
                    f"Protein flexibility missing for pr_id={pr_id}; "
                    f"using fill value {self.protein_flex_fill_value}.",
                    RuntimeWarning,
                )
                self._warned_missing_protein_flex.add(str(pr_id))
            return torch.full((expected_len, 1), self.protein_flex_fill_value, dtype=torch.float32)

        flex = torch.as_tensor(value, dtype=torch.float32)
        if flex.dim() == 1:
            flex = flex.unsqueeze(-1)
        if flex.dim() != 2 or flex.size(-1) != 1:
            raise ValueError(
                f"protein_flex for pr_id={pr_id} must have shape [L_p, 1], "
                f"got {tuple(flex.shape)}."
            )

        if flex.size(0) != expected_len:
            raw_len = flex.size(0)
            # DynaMine returns one value per real residue. Cached ESM features are
            # produced with tokenizer(..., truncation=True, max_length=512), so they
            # include special tokens (<cls>/<eos>) and may also be sequence-truncated.
            # Normal alignments:
            #   raw_len + 2 == expected_len: [CLS=0] + all residues + [EOS=0]
            #   raw_len > expected_len:      [CLS=0] + first expected_len-2 residues + [EOS=0]
            if expected_len >= 2 and raw_len >= expected_len - 2:
                special = torch.full((1, 1), self.protein_flex_fill_value, dtype=torch.float32)
                residue_keep_len = expected_len - 2
                flex = torch.cat([special, flex[:residue_keep_len], special], dim=0)
            elif expected_len == raw_len + 1:
                special = torch.full((1, 1), self.protein_flex_fill_value, dtype=torch.float32)
                flex = torch.cat([flex, special], dim=0)
            else:
                warn_key = str(matched_key)
                if warn_key not in self._warned_shape_protein_flex:
                    warnings.warn(
                        f"Protein flexibility length mismatch for pr_id={pr_id}: "
                        f"flex length={raw_len}, protein feature length={expected_len}. "
                        "Padding/truncating with fill value to keep training running.",
                        RuntimeWarning,
                    )
                    self._warned_shape_protein_flex.add(warn_key)
                aligned = torch.full((expected_len, 1), self.protein_flex_fill_value, dtype=torch.float32)
                keep_len = min(expected_len, raw_len)
                aligned[:keep_len] = flex[:keep_len]
                flex = aligned
        return flex

    def _get_protein_feature(self, pr_id):
        # New format: dict with string keys (supports alphanumeric IDs).
        if isinstance(self.pr_features, dict):
            key = str(pr_id)
            if key in self.pr_features:
                return self.pr_features[key]
            # Backward compatibility: some old dict files may still use int keys.
            if pr_id in self.pr_features:
                return self.pr_features[pr_id]
            try:
                numeric_key = int(pr_id)
                if numeric_key in self.pr_features:
                    return self.pr_features[numeric_key]
            except (TypeError, ValueError):
                pass
            raise KeyError(f"pr_id={pr_id} not found in protein features dict.")

        # Old format: list indexed by numeric pr_id.
        try:
            return self.pr_features[int(pr_id)]
        except (TypeError, ValueError, IndexError):
            raise KeyError(
                f"pr_id={pr_id} cannot index legacy protein feature list. "
                "Please regenerate protein features with dict format."
            )

def randomize_smile(sml):
    """Function that randomizes a SMILES sequence. This was adapted from the
    implemetation of E. Bjerrum 2017, SMILES Enumeration as Data Augmentation
    for Neural Network Modeling of Molecules.
    Args:
        sml: SMILES sequence to randomize.
    Return:
        randomized SMILES sequence or
        nan if SMILES is not interpretable.
    """
    try:
        m = Chem.MolFromSmiles(sml)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        smiles = Chem.MolToSmiles(nm, canonical=False)

        return smiles

    except:
        return sml


def randomize_smile_with_mapping(sml):
    try:
        # Parse the SMILES string into a molecule object
        m = Chem.MolFromSmiles(sml)
        # Create a mapping from atom index to its position in the SMILES
        atom_indices = list(range(m.GetNumAtoms()))
        np.random.shuffle(atom_indices)  # Shuffle the indices

        # Renumber the atoms according to the shuffled indices
        nm = Chem.RenumberAtoms(m, atom_indices)

        # Generate the new SMILES string
        new_smiles = Chem.MolToSmiles(nm, canonical=False)

        # Create a mapping from original atom indices to new positions
        original_to_new_positions = {i: atom_indices.index(i) for i in range(len(atom_indices))}

        return new_smiles, original_to_new_positions
    except:
        return sml, {}

def convert_batch_pr(batch):
    max_len = max([tensor.shape[0] for tensor in batch])
    mask = torch.zeros(len(batch), max_len)
    for i, tensor in enumerate(batch):
        mask[i, :tensor.shape[0]] = 1
    padded_batch = pad(batch, batch_first=True)
    return {'input_ids': padded_batch, 'attention_mask': mask}

def mask_tokens(inputs, attention_mask, tokenizer, probability=0.15):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    mask = attention_mask.clone()
    mask[:, 0] = 0  # cls token
    eos_pos = mask.sum(dim=1).unsqueeze(1)
    mask.scatter_(1, eos_pos, 0)
    mask = mask.bool()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    masked_indices = torch.bernoulli(torch.full(labels.shape, probability)).bool()
    masked_indices = masked_indices * mask
    labels[~masked_indices] = -1  # We only compute loss on masked tokens
    #pos = (labels + 1).nonzero()[;]
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def drop_tokens_with_aux(batch, drop_prob=0.7, aa_avg_f=None, mutation_rate=0, aux_batch=None):
    batch_id, batch_mask = batch['input_ids'], batch['attention_mask']
    # Step 1: Target random deletion. aux_batch, e.g. protein_flex [B, L_p, 1],
    # is sliced with the same indices so it stays aligned with protein tokens/mask.
    if 0 < drop_prob:
        num_to_retain = int(max(1, batch_id.size(1) * (1-drop_prob)))

        indices = sorted(random.sample(range(1, batch_id.size(1)), num_to_retain))
        indices.insert(0, 0)
        batch_id = batch_id[:, indices]
        batch_mask = batch_mask[:, indices]
        if aux_batch is not None:
            aux_batch = aux_batch[:, indices]

    # Step 2: Random mutation changes protein features only; flexibility values remain
    # aligned to residue positions and are therefore not modified.
    if aa_avg_f is not None and mutation_rate > 0:
        # Excluding CLS and EOS positions
        mask = batch_mask.clone()
        mask[:, 0] = 0  # CLS token
        eos_pos = mask.sum(dim=1).unsqueeze(1).to(dtype=torch.int64)
        mask.scatter_(1, eos_pos, 0)  # EOS token
        mask = mask.bool()
        # generate replace mask
        replace_mask = torch.rand(mask.shape) < mutation_rate
        replace_mask &= mask

        # generate random indices
        random_indices = torch.randint(
            0, aa_avg_f.size(0),
            batch_id.shape[:2], device=batch_id.device
        )

        # replace amino acid feature
        replacements = aa_avg_f[random_indices]

        batch_id[replace_mask] = replacements[replace_mask]

    return batch_id, batch_mask, aux_batch


def drop_tokens(batch, drop_prob=0.7, aa_avg_f=None,mutation_rate=0):
    batch_id, batch_mask, _ = drop_tokens_with_aux(
        batch, drop_prob=drop_prob, aa_avg_f=aa_avg_f,
        mutation_rate=mutation_rate, aux_batch=None
    )
    return batch_id, batch_mask

def get_dataLoader(batch_size, dataset, drug_tokenizer,aa=None, shuffle=False, MLM=False,
                   mask_rate=0, target_random_deletion_ratio=0, mutation_rate=0):
    def collate_fn(batch_samples):
        batch_Drug, batch_Protein, batch_seq, batch_label = [], [], [], []
        batch_protein_id, batch_drug_id = [], []
        batch_drug_flex = []
        batch_protein_flex = []
        for sample in batch_samples:
            batch_Drug.append(sample['SMILES'])
            batch_Protein.append(sample['Protein'])
            batch_seq.append(sample['Protein_seq'])
            batch_label.append(sample['Y'])
            batch_protein_id.append(sample['protein_id'])
            batch_drug_id.append(sample['drug_id'])
            if 'drug_flex' in sample:
                batch_drug_flex.append(sample['drug_flex'])
            if 'protein_flex' in sample:
                batch_protein_flex.append(sample['protein_flex'])
        batch_pr = convert_batch_pr(batch_Protein)
        protein_flex = None
        if batch_protein_flex:
            # Shape before deletion: [B, L_p, 1]. Padding positions are ignored by mask.
            protein_flex = pad(batch_protein_flex, batch_first=True)
        batch_inputs_drug = drug_tokenizer(batch_Drug, padding='longest', return_tensors="pt", truncation=True, max_length=200)
        batch_inputs_drug_m, masked_drug_labels = None, None
        # if 0 < target_mask_rate:
        batch_pr['input_ids'], batch_pr['attention_mask'], protein_flex = drop_tokens_with_aux(
            batch_pr,
            target_random_deletion_ratio,
            aa_avg_f=aa,
            mutation_rate=mutation_rate,
            aux_batch=protein_flex,
        )

        if MLM:
            batch_inputs_drug_m = copy.deepcopy(batch_inputs_drug)
            batch_inputs_drug_m['input_ids'], masked_drug_labels\
                = mask_tokens(batch_inputs_drug_m['input_ids'], batch_inputs_drug_m['attention_mask'], drug_tokenizer,mask_rate)

        batch = {
            'batch_inputs_drug': batch_inputs_drug,
            'batch_inputs_drug_m': batch_inputs_drug_m,
            'masked_drug_labels': masked_drug_labels,
            'batch_inputs_pr': batch_pr,
            'labels': batch_label,
            'protein_ids': batch_protein_id,
            'drug_ids': batch_drug_id,
        }
        if batch_drug_flex:
            # Shape: [B, drug_flex_dim], e.g. [B, 1] for log(1 + nTABS).
            batch['batch_drug_flex'] = torch.stack(batch_drug_flex, dim=0)
        if protein_flex is not None:
            # Shape: [B, L_p, 1], same L_p as batch_inputs_pr['input_ids'] after deletion.
            batch['batch_protein_flex'] = protein_flex
        return batch
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
