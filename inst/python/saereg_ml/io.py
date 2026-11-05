"""File exchange with the R package: matrices as Parquet, tables as CSV, a manifest per stage."""

import hashlib
import json
import platform
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import pandas as pd


def write_matrix(path, row_ids, X, id_name, prefix):
    """Write a matrix with one id column and columns prefix0000, prefix0001, ..."""
    X = np.asarray(X)
    cols = {f"{prefix}{j:04d}": X[:, j] for j in range(X.shape[1])}
    pd.DataFrame({id_name: row_ids, **cols}).to_parquet(path, index=False)


def read_matrix(path, id_name):
    """Inverse of write_matrix: (ids, matrix)."""
    df = pd.read_parquet(path)
    return df[id_name].to_numpy(), df.drop(columns=id_name).to_numpy()


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(cfg, stage, outputs, extra=None):
    """Record what a stage wrote, with the config and package versions used."""
    def v(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None
    manifest = {
        "stage": stage,
        "time": datetime.now().isoformat(timespec="seconds"),
        "config": cfg["_config_path"],
        "config_md5": file_md5(cfg["_config_path"]),
        "outputs": [str(o) for o in outputs],
        "versions": {"python": platform.python_version(), "hypothesaes": v("hypothesaes"),
                     "torch": v("torch"), "numpy": v("numpy"), "openai": v("openai")},
        **(extra or {}),
    }
    path = cfg["_run_dir"] / f"manifest_{stage}.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def refuse_overwrite(paths, force):
    existing = [str(p) for p in paths if p.exists()]
    if existing and not force:
        raise FileExistsError(f"Outputs exist (use force=True / --force to overwrite): {existing}")
