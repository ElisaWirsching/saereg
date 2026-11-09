"""Document embeddings: import a cached matrix, or call the OpenAI API.

Decision (2026-09-24, EW): a text longer than the model's input limit is split into
chunks of at most CHUNK_TOKENS tokens; its embedding is the token-weighted mean of the
chunk embeddings, rescaled to unit length. Texts within the limit are sent unchanged, so
earlier runs are unaffected. Requests are batched by token count (API limits).

Decision (2026-09-24, EW): the API checks its per-request limit against UTF-8 bytes / 4,
not the tiktoken count. Evidence: the first Congress request (248,923 tiktoken tokens)
was refused as "Requested 307636 tokens", its bytes / 4. A request stays below
BATCH_TOKENS on both counts.
"""

import os

import numpy as np

from .config import ROOT, load_documents, run_dir
from .io import write_matrix, write_manifest, refuse_overwrite

MAX_TOKENS    = 8191      # input limit of text-embedding-3 models
CHUNK_TOKENS  = 8000      # chunk size for longer texts
BATCH_TOKENS  = 250_000   # tokens per request (API limit 300,000)
BATCH_INPUTS  = 2048      # inputs per request (API limit)


def _chunks(texts, model):
    """(chunk texts, their token counts, document index of each chunk)."""
    import tiktoken
    enc = tiktoken.encoding_for_model(model)
    pieces, n_tok, owner = [], [], []
    for d, t in enumerate(texts):
        tok = enc.encode(t)
        if len(tok) <= MAX_TOKENS:
            pieces.append(t); n_tok.append(len(tok)); owner.append(d)
        else:
            for k in range(0, len(tok), CHUNK_TOKENS):
                part = tok[k:k + CHUNK_TOKENS]
                pieces.append(enc.decode(part)); n_tok.append(len(part)); owner.append(d)
    return pieces, np.array(n_tok), np.array(owner)


def _batches(pieces, n_tok):
    """Index ranges of consecutive chunks within both request limits."""
    size = np.maximum(n_tok, np.ceil([len(p.encode("utf-8")) / 4 for p in pieces]))
    out, start, tot = [], 0, 0
    for i, n in enumerate(size):
        if i > start and (tot + n > BATCH_TOKENS or i - start >= BATCH_INPUTS):
            out.append((start, i)); start, tot = i, 0
        tot += n
    out.append((start, len(n_tok)))
    return out


def embed_texts(cfg, force=False, dry_run=False):
    """Write documents.csv and embeddings.parquet (rows in document order).

    If the config names `embeddings.import_cache`, that .npy matrix is used
    (it must have one row per document, in file order). Otherwise the texts are
    sent to the OpenAI embeddings API, which needs OPENAI_KEY_SAE and costs
    money; `dry_run=True` only reports what would be sent.
    """
    docs = load_documents(cfg)
    out_docs, out_emb = run_dir(cfg, "documents.csv"), run_dir(cfg, "embeddings.parquet")
    src = cfg["embeddings"].get("import_cache")
    model = cfg["embeddings"]["model"]
    dims = cfg["embeddings"].get("dimensions")
    extra = {"dimensions": dims} if dims is not None else {}    # sent only when set
    used = set()                                                # models the API reports

    if src is None and dry_run:
        pieces, n_tok, owner = _chunks(docs["text"].tolist(), model)
        n_long = int(np.sum(np.bincount(owner) > 1))
        print(f"[dry run] would embed {len(docs)} documents ({n_tok.sum():,} tokens; "
              f"{n_long} longer than {MAX_TOKENS} tokens, sent as {len(pieces) - len(docs) + n_long} "
              f"chunks) with {model} in {len(_batches(pieces, n_tok))} requests")
        return None
    refuse_overwrite([out_docs, out_emb], force)

    if src is not None:
        E = np.load(ROOT / src)
        if E.shape[0] != len(docs):
            raise ValueError(f"cache has {E.shape[0]} rows, documents file has {len(docs)}")
        source = f"imported {src}"
    else:
        from openai import OpenAI
        key = os.environ.get("OPENAI_KEY_SAE")
        if not key:
            raise EnvironmentError("Set OPENAI_KEY_SAE to embed through the API.")
        client = OpenAI(api_key=key)
        pieces, n_tok, owner = _chunks(docs["text"].tolist(), model)
        rows = []
        for a, b in _batches(pieces, n_tok):
            resp = client.embeddings.create(input=pieces[a:b], model=model, **extra)
            used.add(resp.model)
            rows.extend(r.embedding for r in resp.data)
        C = np.asarray(rows, dtype=np.float64)
        E = np.zeros((len(docs), C.shape[1]))
        np.add.at(E, owner, C * n_tok[:, None])               # token-weighted sum of chunks
        E /= np.linalg.norm(E, axis=1, keepdims=True)          # unit length (single chunks unchanged)
        E = E.astype(np.float32)
        source = f"OpenAI API ({model}); {int(np.sum(np.bincount(owner) > 1))} texts chunked"

    docs.to_csv(out_docs, index=False)
    write_matrix(out_emb, docs["doc_id"], E, id_name="doc_id", prefix="e")
    write_manifest(cfg, "embed", [out_docs, out_emb],
                   {"source": source, "shape": list(E.shape), "model_used": sorted(used)})
    print(f"embeddings: {E.shape[0]} documents x {E.shape[1]} dims ({source})")
    return E
