"""TopK sparse autoencoder on the training half (hypothesaes.train_sae); its activations."""

import contextlib
import json
import os

import numpy as np
import pandas as pd
import torch

from .config import run_dir
from .io import read_matrix, write_matrix, write_manifest, refuse_overwrite


def load_train(cfg):
    """Training-half document ids, texts and embeddings, in document order."""
    ids, E = read_matrix(run_dir(cfg, "embeddings.parquet"), "doc_id")
    docs = pd.read_csv(run_dir(cfg, "documents.csv"), dtype={"doc_id": str})
    split = pd.read_csv(run_dir(cfg, "split.csv"), dtype={"doc_id": str})
    if not (list(ids) == list(docs["doc_id"]) == list(split["doc_id"])):
        raise ValueError("documents, embeddings and split are not in the same row order")
    tr = np.where(split["is_train"].to_numpy() == 1)[0]
    return ids[tr], docs["text"].to_numpy()[tr], E[tr].astype(np.float32)


def fit_sae(cfg, force=False):
    """Train the SAE on the training half; write sae/sae.pt and sae/decoder.parquet.

    Seeds NumPy and torch immediately before training, which gives the same
    random state as 5g (nothing between its seeding and train_sae draws from
    either generator). decoder.parquet holds unit-norm decoder directions w_j
    as columns f0000..: one row per embedding dimension.
    """
    from hypothesaes.quickstart import train_sae

    out_pt, out_dec = run_dir(cfg, "sae", "sae.pt"), run_dir(cfg, "sae", "decoder.parquet")
    refuse_overwrite([out_pt, out_dec], force)
    _, _, E_tr = load_train(cfg)
    s = cfg["sae"]

    np.random.seed(s["seed"])
    torch.manual_seed(s["seed"])
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        sae = train_sae(embeddings=E_tr, M=s["M"], K=s["K"], checkpoint_dir=None,
                        overwrite_checkpoint=True, show_progress=False)

    W = sae.decoder.weight.detach().cpu().numpy()                    # D x M
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)       # as in 5g
    sae.save(str(out_pt))
    write_matrix(out_dec, np.arange(W.shape[0]), W, id_name="dim", prefix="f")
    meta = {"M": s["M"], "K": s["K"], "seed": s["seed"], "n_train": int(E_tr.shape[0]),
            "device": str(sae.device), "dims": int(W.shape[0])}
    run_dir(cfg, "sae", "meta.json").write_text(json.dumps(meta, indent=2))
    write_manifest(cfg, "sae", [out_pt, out_dec], meta)
    print(f"SAE: M={s['M']}, K={s['K']}, trained on {E_tr.shape[0]} documents ({sae.device})")
    encode(cfg, force=force)
    return sae


def encode(cfg, force=False):
    """Write sae/activations.parquet: the SAE's activations a_ij for every document.

    The encoder is the one trained on the training half. Applying it to a test-half
    document is one forward pass, nothing is refit (as fitNewDocuments() for an STM).
    Rows in document order, columns f0000.. as in decoder.parquet.
    """
    out = run_dir(cfg, "sae", "activations.parquet")
    refuse_overwrite([out], force)
    ids, E = read_matrix(run_dir(cfg, "embeddings.parquet"), "doc_id")
    A = np.asarray(load_sae(cfg).get_activations(E.astype(np.float32), show_progress=False))
    write_matrix(out, ids, A, id_name="doc_id", prefix="f")
    meta = {"shape": list(A.shape), "share_active": float((A > 0).mean())}
    write_manifest(cfg, "encode", [out], meta)
    print(f"activations: {A.shape[0]} documents x {A.shape[1]} features "
          f"(on in {meta['share_active']:.3f} of cells)")
    return A


def load_sae(cfg):
    """Reload the trained SAE on the default device (GPU if available).

    The saved config records the training device, and get_activations() sends
    inputs there, so the weights must be loaded onto the same default device.
    """
    from hypothesaes.sae import load_model
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        return load_model(str(run_dir(cfg, "sae", "sae.pt", create=False)))
