"""Feature labels: GPT names what the top-scoring training texts on a decoder direction do.

For feature j, the training-half texts are ranked by their score on the decoder
direction, s_ij = e_i' w_j (the quantity alpha_j projects on). The labeller sees
n_per_bin random texts from each percentile bin in `bins` (bottom 10%, middle 45-55%,
top 10%), lowest first, and writes one label for the top bin: what its texts do that
the texts in the lower bins do not. The lower bins are the contrast; they are not
labelled.

Decision (2026-09-24, EW): three bins of s_ij, one label for the top bin only. Evidence:
  - The bins separate the feature being on from it being off. Of the texts in the top
    10% of s_ij, a median 92% have the feature active (range 68-98% across features); in
    the middle bin 5%; in the bottom bin 0% for every feature (Congress run
    run_imm_large_M20_K4, training half, ad hoc check).
  - The bottom bin is not labelled. The score splits exactly as
    s_ij = w_j'b_pre + a_ij + sum_{k!=j} (w_j'w_k) a_ik + w_j'r_i (method memo, appendix:
    "The score on a decoder direction"); since a_ij >= 0, the low end of s_j is made of
    other features pointing away, which have labels of their own. Its texts are off, so
    they serve as contrast.
  - Superseded the same day: a forced "<high end>, rather than <low end>" (the low-end
    clause named one strand of a mixed group: Congress F4); an optional low end; and texts
    where the feature is on (by activation) against inactive middle texts only. On the
    Egami main run the top-only bins and the on/off design gave the same labels for the
    three overlapping features F8, F9, F5 (all prior offenses) and close labels elsewhere;
    the bins rank by s_ij, the quantity alpha_j is built from, and need no SAE activations.
  - Overlapping features get similar labels in every design (Egami main F8, F9, F5:
    decoder cosines 0.22-0.24; F8 is also active in 60% of F5's texts). Their overlap is
    carried by the joint share R^2_S, not by the labels.
Earlier decisions: replaces hypothesaes.interpret_sae() (it contrasts the 10 top-activating
texts, the most extreme cases, with random zero-activation texts); terciles of s_ij gave
near-identical labels (Egami main run: |corr| of s_j with PC1 or PC2 up to 0.92), so the
bins sit at the ends.
"""

import os
import re

import numpy as np
import pandas as pd

from .config import run_dir
from .io import read_matrix, write_manifest, refuse_overwrite
from .sae import load_train

PROMPT = """You are a machine learning researcher studying one direction in a text-embedding space. Every text has a score on this direction.
Below, texts are grouped by their score, from the lowest scores (Group 1) to the highest (Group {n_bins}). Each group shows randomly chosen texts from its range.
Your goal is to describe what the texts with the highest scores (Group {n_bins}) do that the other texts do not. Use the lower groups to tell which features of Group {n_bins} are distinctive and which are common to all texts.

{groups}

Rules for your label:
- Describe a concrete, objective feature of the texts (what they say, argue or propose, or how they say it), not an abstract concept.
- The feature must fit most texts in Group {n_bins}, not only one or two of them, and be rarer in the lower groups. Do not describe something that all groups share.
- Use at most {max_label_words} words.

Example labels from other studies:
- "praises the taste of the food"
- "gives specific numbers and dates"

Answer in exactly this format and nothing else:
LABEL: "<label>"
"""


def _cut(text, max_words):
    words = str(text).split()
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")


def _bin_members(s, bins):
    """Indices of the texts in each percentile bin of the scores s, lowest bin first."""
    if len(bins) < 2 or any(b[0] >= b[1] for b in bins) or \
            any(bins[g][1] > bins[g + 1][0] for g in range(len(bins) - 1)):
        raise ValueError("labels.bins must be at least two increasing, non-overlapping "
                         "percentile ranges, lowest first")
    out = []
    for lo, hi in bins:
        q_lo, q_hi = np.percentile(s, [lo, hi])
        out.append(np.where((s >= q_lo) & (s <= q_hi))[0])
    return out


def _shown(members, lab, j):
    """The texts the labeller sees for feature j: n_per_bin random members of each bin."""
    rng = np.random.default_rng(lab["seed"] + j)
    return [rng.choice(idx, size=min(lab["n_per_bin"], len(idx)), replace=False)
            for idx in members]


def _train_scores(cfg):
    """Training-half ids, texts and scores s_ij = e_i' w_j (documents x features)."""
    ids, texts, E = load_train(cfg)
    _, W = read_matrix(run_dir(cfg, "sae", "decoder.parquet", create=False), "dim")  # D x M
    return ids, texts, E @ W


def build_prompts(cfg, texts=None, E=None, W=None):
    """One prompt per feature, from the training half (list of strings).

    By default the texts, embeddings and decoder come from the run directory; pass
    texts, E (documents x dims) and W (dims x features) to label other directions on
    other documents (e.g. SAEs trained inside a simulation).
    """
    lab = cfg["labels"]
    bins = lab["bins"]
    if texts is None:
        _, texts, S = _train_scores(cfg)
    else:
        S = E @ W                                                                    # scores s_ij
    prompts = []
    for j in range(S.shape[1]):
        members = _bin_members(S[:, j], bins)
        groups = []
        for g, ((lo, hi), idx, pick) in enumerate(zip(bins, members, _shown(members, lab, j))):
            groups.append(f"GROUP {g + 1} (score percentiles {lo:.0f}-{hi:.0f}; "
                          f"{len(pick)} of {len(idx)} texts shown):\n" +
                          "\n".join(f"- {_cut(texts[i], lab['max_words'])}" for i in pick))
        # filled literally, so other braces are left alone; the texts go last so that a
        # placeholder inside a text stays as written
        prompts.append((lab.get("prompt") or PROMPT)
                       .replace("{n_bins}", str(len(bins)))
                       .replace("{max_label_words}", str(lab["max_label_words"]))
                       .replace("{groups}", "\n\n".join(groups)))
    return prompts


def label_features(cfg, force=False, dry_run=False):
    """Write labels/labels.csv (feature, label).

    Calls the OpenAI API (needs OPENAI_KEY_SAE), one request per feature.
    `dry_run=True` writes the exact prompts to labels/prompts_dryrun/ instead.
    Settings in cfg["labels"]: model, temperature, bins (percentile ranges of s_ij,
    lowest first; the label describes the last), n_per_bin, max_words (per text shown),
    max_label_words (asked for, not enforced; longer labels are reported), seed (text
    sampling, seed + j for feature j).
    """
    lab = cfg["labels"]
    prompts = build_prompts(cfg)
    if dry_run:
        out = run_dir(cfg, "labels", "prompts_dryrun", "x.txt").parent
        for j, p in enumerate(prompts):
            (out / f"feature_{j}.txt").write_text(p)
        print(f"[dry run] wrote {len(prompts)} prompts to {out}; model {lab['model']}")
        return None

    out = run_dir(cfg, "labels", "labels.csv")
    refuse_overwrite([out], force)
    used = []
    labels = pd.DataFrame({"feature": range(len(prompts)),
                           "label": ask_labeller(prompts, lab, used)})
    labels.to_csv(out, index=False)
    write_manifest(cfg, "label", [out], {"labels": {k: lab[k] for k in lab}, "model_used": used})
    print(labels.to_string(index=False))
    over = labels["label"].str.split().str.len() > int(lab["max_label_words"])
    if over.any():
        print(f"[note] {int(over.sum())} of {len(labels)} labels exceed {lab['max_label_words']} "
              f"words: features {labels.loc[over, 'feature'].tolist()}")
    return labels


def ask_labeller(prompts, lab, used=None):
    """Send each prompt to the labelling model; return the labels (list of strings).

    `used`, if given, is a list that receives the model the API reports for each prompt.

    Calls the OpenAI API (needs OPENAI_KEY_SAE). A reply without a LABEL line is kept
    whole, with a note.
    """
    key = os.environ.get("OPENAI_KEY_SAE")
    if not key:
        raise EnvironmentError("Set OPENAI_KEY_SAE to label features through the API.")
    from openai import OpenAI
    client = OpenAI(api_key=key)
    labels = []
    for j, p in enumerate(prompts):
        r = client.chat.completions.create(model=lab["model"], temperature=lab["temperature"],
                                           messages=[{"role": "user", "content": p}])
        if used is not None:
            used.append(r.model)
        text = r.choices[0].message.content.strip()
        m = re.search(r'LABEL:\s*"?(.*?)"?\s*$', text, flags=re.S)
        if m is None:
            print(f"[note] feature {j}: no LABEL line; the whole reply is kept as its label")
        labels.append(m.group(1).strip() if m else text)
    return labels


def shown_texts(cfg, force=False):
    """Write labels/shown_texts.csv: the training texts the labeller was shown.

    One row per text and feature, in prompt order: feature, bin (1 = lowest), the bin's
    percentile range and score bounds (as the labeller computed them), position in the
    prompt, doc_id and score s_ij. The same seeded draw as build_prompts(); each row is
    checked against the prompt. No API. Added 2026-09-25 (EW) to show the labeller's evidence
    in figures.
    """
    out = run_dir(cfg, "labels", "shown_texts.csv")
    refuse_overwrite([out], force)
    lab = cfg["labels"]
    bins = lab["bins"]
    ids, texts, S = _train_scores(cfg)
    prompts = build_prompts(cfg)
    rows = []
    for j in range(S.shape[1]):
        members = _bin_members(S[:, j], bins)
        for g, ((lo, hi), pick) in enumerate(zip(bins, _shown(members, lab, j))):
            q_lo, q_hi = np.percentile(S[:, j], [lo, hi])
            for pos, i in enumerate(pick):
                if _cut(texts[i], lab["max_words"]) not in prompts[j]:
                    raise AssertionError(f"feature {j}: a shown text is not in its prompt")
                rows.append({"feature": j, "bin": g + 1, "pct_lo": lo, "pct_hi": hi,
                             "score_lo": q_lo, "score_hi": q_hi, "position": pos + 1,
                             "doc_id": ids[i], "score": S[i, j]})
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    write_manifest(cfg, "shown", [out])
    print(f"{len(df)} shown texts ({S.shape[1]} features) written to {out}")
    return df


# ---- Label fidelity -------------------------------------------------------------------
# Decision (2026-09-25, EW): does a label pick out its feature's texts on texts the labeller
# did not see? A second model reads each text with the label only and answers YES or NO;
# F1 against the text's bin (as Movva et al. 2025, App. C.1, with the bins of s_ij in place
# of activation deciles). Texts: n_top from the top bin and n_lower from the lower bins
# pooled (split evenly), training half, excluding the texts shown to the labeller. No
# cross-feature matrix. Checks the labeller, not whether the feature matches a truth.

FIDELITY_PROMPT = """You will read a text and a description of a feature that texts can have.

Feature: "{label}"

Text: "{text}"

Does the text have this feature? Answer with exactly one word: YES or NO."""


def fidelity_sample(cfg, S):
    """Texts for the fidelity test: DataFrame(feature, row, group, bin), rows of S."""
    lab, fid = cfg["labels"], cfg["fidelity"]
    bins = lab["bins"]
    per_lower = fid["n_lower"] // (len(bins) - 1)
    rows = []
    for j in range(S.shape[1]):
        members = _bin_members(S[:, j], bins)
        shown = _shown(members, lab, j)
        rng = np.random.default_rng(fid["seed"] + j)
        for g, (idx, seen) in enumerate(zip(members, shown)):
            pool = np.setdiff1d(idx, seen)
            size = fid["n_top"] if g == len(bins) - 1 else per_lower
            pick = rng.choice(pool, size=min(size, len(pool)), replace=False)
            group = "top" if g == len(bins) - 1 else "lower"
            rows += [(j, int(i), group, g + 1) for i in pick]
    return pd.DataFrame(rows, columns=["feature", "row", "group", "bin"])


def label_fidelity(cfg, force=False, dry_run=False, labels=None, suffix="", data=None):
    """Write labels/fidelity_annotations.csv (one row per text read) and
    labels/label_fidelity.csv (per feature: YES counts, precision, recall, F1).

    Calls the OpenAI API (needs OPENAI_KEY_SAE), one request per text. `dry_run=True`
    writes the requests to labels/fidelity_prompts_dryrun.csv instead. Settings in
    cfg["fidelity"]: model, temperature, n_top, n_lower, max_words (per text), seed;
    the bins and the texts shown to the labeller come from cfg["labels"].
    Precision = YES in the top bin / all YES; recall = YES in the top bin / texts read
    from the top bin; F1 their harmonic mean.
    `labels` (DataFrame feature, label) tests another set of labels, e.g. a human's, on
    the same texts; `suffix` is added to the output file names (added 2026-09-29).
    `data` (ids, texts, S: documents x features) reads other documents and scores in place
    of the run directory's training half, as build_prompts' texts, E and W do (added
    2026-10-11 for the Congress tweet SAEs, whose embeddings live outside a run directory).
    """
    fid = cfg["fidelity"]
    if labels is None:
        labels = pd.read_csv(run_dir(cfg, "labels", "labels.csv", create=False))
    labels = labels.reset_index(drop=True)
    ids, texts, S = _train_scores(cfg) if data is None else data
    if list(labels["feature"]) != list(range(S.shape[1])):
        raise ValueError("labels.csv does not have one label per feature, in order")
    smp = fidelity_sample(cfg, S)
    smp["doc_id"] = ids[smp["row"]]
    prompts = [FIDELITY_PROMPT.format(label=labels["label"][j],
                                      text=_cut(texts[i], fid["max_words"]))
               for j, i in zip(smp["feature"], smp["row"])]
    if dry_run:
        out = run_dir(cfg, "labels", f"fidelity_prompts_dryrun{suffix}.csv")
        smp.drop(columns="row").assign(prompt=prompts).to_csv(out, index=False)
        n_words = sum(len(p.split()) for p in prompts)
        print(f"[dry run] wrote {len(prompts)} requests ({n_words} words) to {out}; "
              f"model {fid['model']}")
        return None

    out_a = run_dir(cfg, "labels", f"fidelity_annotations{suffix}.csv")
    out_f = run_dir(cfg, "labels", f"label_fidelity{suffix}.csv")
    refuse_overwrite([out_a, out_f], force)
    used = []
    ann = smp.drop(columns="row").assign(answer=ask_reader(prompts, fid, used))
    ann.to_csv(out_a, index=False)
    res = []
    for j, a in ann.groupby("feature"):
        yes_top = int(((a["group"] == "top") & (a["answer"] == "YES")).sum())
        yes_low = int(((a["group"] == "lower") & (a["answer"] == "YES")).sum())
        n_top = int((a["group"] == "top").sum())
        prec = yes_top / (yes_top + yes_low) if yes_top + yes_low > 0 else np.nan
        rec = yes_top / n_top
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        res.append({"feature": j, "label": labels["label"][j], "n_top": n_top,
                    "n_lower": int((a["group"] == "lower").sum()), "yes_top": yes_top,
                    "yes_lower": yes_low, "no_answer": int(a["answer"].isna().sum()),
                    "precision": prec, "recall": rec, "f1": f1})
    res = pd.DataFrame(res)
    res.to_csv(out_f, index=False)
    write_manifest(cfg, f"fidelity{suffix}", [out_a, out_f],
                   {"fidelity": {k: fid[k] for k in fid}, "model_used": sorted(set(used))})
    print(res.drop(columns="label").to_string(index=False))
    return res


def ask_reader(prompts, fid, used=None):
    """Send each fidelity request to the reading model; return "YES", "NO" or None.

    `used`, if given, is a list that receives the model the API reports for each request.

    Calls the OpenAI API (needs OPENAI_KEY_SAE). A reply that is neither is kept as None,
    with a count at the end.
    """
    key = os.environ.get("OPENAI_KEY_SAE")
    if not key:
        raise EnvironmentError("Set OPENAI_KEY_SAE to run the fidelity test through the API.")
    from openai import OpenAI
    client = OpenAI(api_key=key)
    answers = []
    for k, p in enumerate(prompts):
        r = client.chat.completions.create(model=fid["model"], temperature=fid["temperature"],
                                           messages=[{"role": "user", "content": p}])
        if used is not None:
            used.append(r.model)
        text = r.choices[0].message.content.strip().upper()
        answers.append("YES" if text.startswith("YES") else "NO" if text.startswith("NO")
                       else None)
        if (k + 1) % 200 == 0:
            print(f"  {k + 1} of {len(prompts)} read")
    n_none = sum(a is None for a in answers)
    if n_none:
        print(f"[note] {n_none} replies were neither YES nor NO; kept as missing")
    return answers
