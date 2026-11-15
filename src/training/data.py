"""Tensor bundle over the corpus + split/domain index helpers."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from ..data.schema import DATASETS, DATASET_ID

SPLITS = {"train": 0, "val": 1, "test": 2}


class Bundle:
    """The corpus as tensors. `idx()` returns row indices for a split / domain / dataset selection."""

    def __init__(self, corpus: Dict[str, np.ndarray], meta: Dict):
        self.c, self.meta = corpus, meta
        self.X = torch.from_numpy(corpus["X"]).float()
        self.rul = torch.from_numpy(corpus["rul"]).float()
        self.has_rul = torch.from_numpy(corpus["has_rul"].astype(bool))
        self.event = torch.from_numpy(corpus["event"].astype(np.float32))
        self.fail = torch.from_numpy(corpus["fail"]).float()
        self.domain = corpus["domain"].astype(int)
        self.split = corpus["split"].astype(int)
        self.dataset = corpus["dataset"].astype(int)
        self.label_avail = corpus["label_avail"].astype(bool)
        self.n_gaps = corpus["n_gaps"].astype(int)
        self.group = corpus["group"]
        self.n, self.T, self.C = self.X.shape

    def idx(self, split: Optional[str] = None, domain: Optional[int] = None,
            datasets: Optional[Sequence[str]] = None, labelled: Optional[bool] = None) -> np.ndarray:
        m = np.ones(self.n, dtype=bool)
        if split is not None:
            m &= self.split == SPLITS[split]
        if domain is not None:
            m &= self.domain == domain
        if datasets is not None:
            m &= np.isin(self.dataset, [DATASET_ID[d] for d in datasets])
        if labelled is not None:
            m &= self.label_avail == labelled
        return np.flatnonzero(m)

    def present_datasets(self) -> list:
        return [d for d in DATASETS if (self.dataset == DATASET_ID[d]).any()]
