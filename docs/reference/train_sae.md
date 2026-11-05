<div id="main" class="col-md-9" role="main">

# Train a sparse autoencoder on embeddings

<div class="ref-description section level2">

A TopK SAE (`hypothesaes`, Python stage `sae`) trained on the rows
marked `train`; the trained encoder then gives the activations of every
row. NumPy and torch are seeded with `seed` right before training, so a
run on the same device reproduces the SAE.

</div>

<div class="section level2">

## Usage

<div class="sourceCode">

``` r
train_sae(
  E,
  M,
  K,
  seed = 7L,
  train = rep(TRUE, nrow(E)),
  dir = tempfile("saereg_sae_")
)
```

</div>

</div>

<div class="section level2">

## Arguments

-   E:

    Embeddings (documents x dimensions).

-   M:

    Number of features (decoder directions).

-   K:

    Features active per document.

-   seed:

    Seed of the training.

-   train:

    Logical, the rows the SAE is trained on (e.g. the training half).
    The others are only encoded.

-   dir:

    Run directory (a new temporary one by default); it keeps
    `sae/sae.pt`.

</div>

<div class="section level2">

## Value

A list: `W` (dimensions x M, unit-length decoder directions),
`activations` (documents x M), `dir`.

</div>

</div>
