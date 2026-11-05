<div id="main" class="col-md-9" role="main">

# Label SAE directions with a large language model

<div class="ref-description section level2">

For each direction w\_j, the texts are ranked by their score e\_i'w\_j;
the model sees `n_per_bin` random texts from the bottom 10%, the middle
45-55% and the top 10% and names what the top texts do that the others
do not (Python stage `label`). With `fidelity = TRUE`, a second call per
text checks whether each label picks out its direction's top texts among
texts the labeller did not see (stage `fidelity`: precision, recall,
F1). Pass the training texts: the labels must not use the texts on which
the effects are estimated.

</div>

<div class="section level2">

## Usage

<div class="sourceCode">

``` r
label_features(
  texts,
  E,
  W,
  fidelity = FALSE,
  dry_run = FALSE,
  model = "gpt-4.1",
  seed = 7L,
  settings = list(),
  dir = tempfile("saereg_label_")
)
```

</div>

</div>

<div class="section level2">

## Arguments

-   texts:

    Character vector (the training texts).

-   E:

    Their embeddings (documents x dimensions).

-   W:

    Decoder directions (dimensions x M).

-   fidelity:

    Also run the fidelity test (100 API calls per direction by default)?

-   dry\_run:

    `TRUE` returns the prompts without calling the API.

-   model, seed:

    Labelling model and seed of the text sampling (direction j: seed +
    j).

-   settings:

    Optional list overriding the defaults: `bins`, `n_per_bin` (7),
    `max_words` (256), `max_label_words` (15), `temperature` (0),
    `n_top` and `n_lower` (50 each, fidelity).

-   dir:

    Run directory (a new temporary one by default).

</div>

<div class="section level2">

## Value

A tibble with `feature` (from 0) and `label`, plus `precision`,
`recall`, `f1` with `fidelity = TRUE`; on a dry run, the prompts
(character vector).

</div>

</div>
