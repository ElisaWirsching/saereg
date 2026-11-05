"""Text and machine-learning stages of the embedding-regression + SAE pipeline.

Stages (each writes into the run directory named in the config):
    embed_texts    documents.csv, embeddings.parquet
    make_split     split.csv
    fit_sae        sae/sae.pt, sae/decoder.parquet (then encode)
    encode         sae/activations.parquet
    label_features labels/labels.csv
    label_fidelity labels/label_fidelity.csv, labels/fidelity_annotations.csv
    shown_texts    labels/shown_texts.csv (the texts the labeller saw)
Estimation and inference are in the R package (Code/pipeline/R/saereg).
"""

from .config import load_config, load_documents
from .embed import embed_texts
from .split import make_split
from .sae import fit_sae, load_sae, encode
from .label import label_features, label_fidelity, shown_texts

__all__ = ["load_config", "load_documents", "embed_texts", "make_split",
           "fit_sae", "load_sae", "encode", "label_features", "label_fidelity",
           "shown_texts"]
