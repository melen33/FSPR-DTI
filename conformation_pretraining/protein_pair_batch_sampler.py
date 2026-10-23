"""Deterministic structural batches that preserve same-protein drug pairs."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import Sampler


class ProteinPairBatchSampler(Sampler):
    """Yield each index once while grouping two same-protein rows in a batch."""

    def __init__(self, protein_ids, batch_size: int, seed: int = 6, group_ids=None, pair_drug_ids=None, drop_unpaired: bool = False, max_distinct_drugs_per_batch: int = 2):
        if int(batch_size) < 2:
            raise ValueError("Structural pair batches require batch_size >= 2")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        if group_ids is not None and len(group_ids) != len(protein_ids):
            raise ValueError("group_ids length must match protein_ids")
        if pair_drug_ids is not None and len(pair_drug_ids) != len(protein_ids):
            raise ValueError("pair_drug_ids length must match protein_ids")
        self._pair_drug_ids = (
            None if pair_drug_ids is None else tuple(str(value) for value in pair_drug_ids)
        )
        self.drop_unpaired = bool(drop_unpaired)
        self.max_distinct_drugs_per_batch = int(max_distinct_drugs_per_batch)
        if self.max_distinct_drugs_per_batch < 2:
            raise ValueError("max_distinct_drugs_per_batch must be at least 2")
        if self.max_distinct_drugs_per_batch > self.batch_size:
            raise ValueError("max_distinct_drugs_per_batch cannot exceed batch_size")
        if self.drop_unpaired and self._pair_drug_ids is None:
            raise ValueError("drop_unpaired requires pair_drug_ids")
        self._groups = defaultdict(list)
        for index, protein_id in enumerate(protein_ids):
            key = str(protein_id) if group_ids is None else str(group_ids[index])
            self._groups[key].append(index)
        if not self._groups:
            raise ValueError("protein_ids must not be empty")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _chunks(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        pairs, singletons = [], []
        multi_candidate = (
            self._pair_drug_ids is not None
            and self.max_distinct_drugs_per_batch > 2
        )
        for indices in self._groups.values():
            permutation = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[position] for position in permutation]
            if self._pair_drug_ids is None:
                for start in range(0, len(shuffled) - 1, 2):
                    pairs.append(shuffled[start:start + 2])
                if len(shuffled) % 2:
                    singletons.append([shuffled[-1]])
                continue
            by_drug = defaultdict(list)
            for index in shuffled:
                by_drug[self._pair_drug_ids[index]].append(index)
            while True:
                active = [drug for drug, values in by_drug.items() if values]
                if len(active) < 2:
                    break
                active.sort(key=lambda drug: len(by_drug[drug]), reverse=True)
                selected = active[:(
                    min(self.max_distinct_drugs_per_batch, len(active))
                    if multi_candidate else 2
                )]
                pairs.append([by_drug[drug].pop() for drug in selected])
            for values in by_drug.values():
                singletons.extend([index] for index in values)
        for chunks in (pairs, singletons):
            if len(chunks) > 1:
                order = torch.randperm(len(chunks), generator=generator).tolist()
                chunks[:] = [chunks[position] for position in order]
        return pairs, singletons

    def __iter__(self):
        pairs, singletons = self._chunks()
        if self._pair_drug_ids is not None and self.max_distinct_drugs_per_batch > 2:
            yield from pairs
        else:
            pair_capacity = max(self.batch_size // 2, 1)
            for start in range(0, len(pairs), pair_capacity):
                yield [index for pair in pairs[start:start + pair_capacity] for index in pair]
        if not self.drop_unpaired:
            for start in range(0, len(singletons), self.batch_size):
                yield [index for item in singletons[start:start + self.batch_size] for index in item]

    def __len__(self):
        pairs, singletons = self._chunks()
        if self._pair_drug_ids is not None and self.max_distinct_drugs_per_batch > 2:
            pair_batches = len(pairs)
        else:
            pair_capacity = max(self.batch_size // 2, 1)
            pair_batches = (len(pairs) + pair_capacity - 1) // pair_capacity
        if self.drop_unpaired:
            return pair_batches
        return pair_batches + (len(singletons) + self.batch_size - 1) // self.batch_size
