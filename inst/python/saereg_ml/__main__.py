"""Command line: python -m saereg_ml <stage> --config <yaml> [--force] [--dry-run]"""

import argparse

from . import (load_config, embed_texts, make_split, fit_sae, encode, label_features,
               label_fidelity, shown_texts)

STAGES = {"embed": embed_texts, "split": make_split, "sae": fit_sae, "encode": encode,
          "label": label_features, "fidelity": label_fidelity, "shown": shown_texts}


def main():
    ap = argparse.ArgumentParser(prog="saereg_ml")
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    ap.add_argument("--dry-run", action="store_true", help="embed/label/fidelity: no API calls")
    a = ap.parse_args()
    cfg = load_config(a.config)
    kwargs = {"force": a.force}
    if a.stage in ("embed", "label", "fidelity"):
        kwargs["dry_run"] = a.dry_run
    STAGES[a.stage](cfg, **kwargs)


if __name__ == "__main__":
    main()
