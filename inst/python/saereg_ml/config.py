"""Load a pipeline config and the documents it points to."""

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path.home() / "Dropbox" / "Research" / "EmbeddingSAE"   # same on Linux and Mac


def load_config(path):
    """Read a YAML config; paths in it are relative to the project root."""
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    cfg["_run_dir"] = ROOT / cfg["run_dir"]
    return cfg


def run_dir(cfg, *parts, create=True):
    """Path inside the run directory (created on demand)."""
    p = Path(cfg["_run_dir"]).joinpath(*parts)
    if create:
        (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
    return p


def load_documents(cfg):
    """Documents in file order with doc_id, cell, text.

    Every row must fall into exactly one configured cell; the cell order in
    the config is kept as the category order.
    """
    d = cfg["documents"]
    raw = pd.read_csv(ROOT / d["file"])
    cell = pd.Series(pd.NA, index=raw.index, dtype="object")
    for name, rule in d["cells"].items():
        m = pd.Series(True, index=raw.index)
        for col, val in rule.items():
            m &= raw[col] == val
        if (m & cell.notna()).any():
            raise ValueError(f"cell '{name}' overlaps an earlier cell")
        cell[m] = name
    if cell.isna().any():
        raise ValueError(f"{int(cell.isna().sum())} documents match no configured cell")
    out = pd.DataFrame({"doc_id": raw[d["id"]].astype(str), "cell": cell.astype(str),
                        "text": raw[d["text"]].astype(str)})
    if out["doc_id"].duplicated().any():
        raise ValueError("document ids are not unique")
    return out
