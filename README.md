# saereg: SAE regression

Treatments and group differences often change texts in several ways at once. **SAE regression**
estimates such an effect on texts and decomposes it into interpretable directions:

1. **Embed** the texts with a language model (`embed_texts()`).
2. **Learn directions** with a sparse autoencoder (SAE) trained on one half of the texts
   (`train_sae()`).
3. **Label** the directions with a large language model, from the training texts that score
   highest on them (`label_features()`).
4. **Estimate and decompose** on the other half (`saereg()`): the effect on the average text,
   a global test, and, for each direction, how far the texts move towards or away from it and
   what share of the effect it explains.

`saereg()` works like a regression. Covariates, absorbed fixed effects and clustered designs
(e.g. difference-in-differences) are supported.

## Installation

```r
# install.packages("remotes")
remotes::install_github("ElisaWirsching/saereg")
```

`saereg()` needs only R. Steps 1 to 3 run the package's Python stages
(`inst/python/saereg_ml`) and need a Python with `numpy`, `pandas`, `pyarrow`, `pyyaml`,
`torch`, `hypothesaes` (version 2.0.0), `tiktoken` and `openai`:

`saereg` looks for such a Python by itself; `saereg_python()` shows which one it uses. To choose
another: `options(saereg.python = "/path/to/python3")`, or the environment variable
`SAEREG_PYTHON`.

Embedding and labeling call the OpenAI API: they need a key in the environment variable
`OPENAI_KEY_SAE` and cost money. With `dry_run = TRUE` they show what they would send without
sending it.

## Example

The package ships the open-ended answers of a survey experiment on immigration and punishment
(Egami et al. 2022), with their embeddings and an SAE trained on the training half.

```r
library(saereg)
d <- egami$data
E <- egami$E
test <- !d$train

fit <- saereg(E[test, ] ~ treat, data = d[test, ], W = egami$W, labels = egami$labels)
fit              # the overall effect: squared norm, bias-corrected, global test, R^2
summary(fit)     # the directions by their share of the effect
confint(fit)     # effects along the directions, with jackknife intervals
```

The [getting-started article](https://github.com/ElisaWirsching/saereg/blob/main/articles/getting_started.md) walks through all four steps, with their outputs; in R, `vignette("getting_started", package = "saereg")`.

## Citation

Wirsching, E. M. (2026). What moved? Interpretable decompositions of effects on text with
sparse autoencoders. Working paper.

`train_sae()` uses the SAE implementation in [HypotheSAEs](https://github.com/rmovva/HypotheSAEs),
and the label-fidelity check in `label_features()` follows the same paper. If you use them, please
also cite Movva, R., Peng, K., Garg, N., Kleinberg, J., and Pierson, E. (2025). Sparse autoencoders
for hypothesis generation. *Proceedings of the 42nd International Conference on Machine Learning*.

## Licence and data

GPL-3. The example data come from the replication data of Egami, Fong, Grimmer, Roberts and
Stewart (2022), "How to make causal inferences using texts", *Science Advances* 8, Harvard
Dataverse, doi:10.7910/DVN/QHJN8V (CC0 1.0).
