"""Stratified train/test split by cell (the Egami et al. 2022 split)."""

import numpy as np
import pandas as pd

from .config import run_dir
from .io import write_manifest, refuse_overwrite


def make_split(cfg, force=False):
    """Write split.csv: doc_id, cell, is_train (1 = train half).

    Within each cell, in config order, one generator draws a permutation and
    the first floor(n * train_share) documents go to the training half. With
    seed 7 and share 0.5 this reproduces the split of 5g_seed7_table.py.
    """
    docs = pd.read_csv(run_dir(cfg, "documents.csv"), dtype={"doc_id": str})
    out = run_dir(cfg, "split.csv")
    refuse_overwrite([out], force)

    s = cfg["split"]
    rng = np.random.default_rng(s["seed"])
    is_train = np.zeros(len(docs), dtype=int)
    for cell in cfg["documents"]["cells"]:                  # config order
        idx = rng.permutation(np.where(docs["cell"].to_numpy() == cell)[0])
        is_train[idx[:int(np.floor(len(idx) * s["train_share"]))]] = 1

    split = pd.DataFrame({"doc_id": docs["doc_id"], "cell": docs["cell"], "is_train": is_train})
    split.to_csv(out, index=False)
    write_manifest(cfg, "split", [out], {"seed": s["seed"], "train_share": s["train_share"]})
    print(split.groupby(["cell", "is_train"]).size().unstack().rename(columns={0: "test", 1: "train"}))
    return split
