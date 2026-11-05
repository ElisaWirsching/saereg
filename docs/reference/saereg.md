<div id="main" class="col-md-9" role="main">

# SAE regression: an embedding regression decomposed onto SAE directions

<div class="ref-description section level2">

Regresses the document embeddings on a design (OLS, covariates and
absorbed fixed effects allowed), takes the coefficient vector of the
term of interest, beta, as the effect on the texts, tests it globally,
and decomposes it onto the decoder directions W of a sparse autoencoder
trained on other texts. Every quantity is linear in the documents: beta
= t(E) C, where C, the term's row weights, comes from the design alone
(Frisch-Waugh-Lovell).

</div>

<div class="section level2">

## Usage

<div class="sourceCode">

``` r
saereg(
  formula,
  data,
  W,
  term = NULL,
  cluster = NULL,
  strata = NULL,
  test = c("permutation", "bootstrap"),
  draws = 999L,
  seed = 7L,
  level = 0.95,
  activations = NULL,
  labels = NULL,
  rows = NULL,
  cores = 1L
)
```

</div>

</div>

<div class="section level2">

## Arguments

-   formula:

    `E ~ treat + covariates | fe1 + fe2`. The left side is the embedding
    matrix (documents x dimensions), found in `data` or in the formula's
    environment. Fixed effects after `|` are absorbed
    (`fixest::demean()`); combine variables beforehand for interacted
    effects (e.g. member x stack).

-   data:

    Data frame with the variables. Rows with a missing value are
    dropped.

-   W:

    Decoder directions (dimensions x M), unit-length columns, from an
    SAE trained on texts not used here.

-   term:

    Column(s) of the design whose effect is decomposed; default the
    first variable on the right side. Several terms (e.g. party-specific
    treatment indicators) are estimated jointly and decomposed one by
    one.

-   cluster:

    One-sided formula, e.g. `~ member`: jackknife and bootstrap by
    cluster. Required with fixed effects or `rows`.

-   strata:

    One-sided formula, constant within clusters: the bootstrap resamples
    within each stratum (e.g. ever-treated and never-treated members).

-   test:

    `"permutation"` or `"bootstrap"`.

-   draws:

    Permutations or bootstrap draws.

-   seed:

    Permutations: `set.seed(seed)` once, then one draw after another.
    Bootstrap: draw b uses `set.seed(seed + b)`, so any draw can be
    re-run on its own.

-   level:

    Confidence level of the intervals.

-   activations:

    Optional SAE activations (documents x M, rows as `E`).

-   labels:

    Optional labels of the M directions.

-   rows:

    Optional row of `E` for each row of `data`, when rows share an
    embedding (e.g. a member-day's mean in several stacks); their
    weights are summed per embedding.

-   cores:

    Cores for the jackknife and bootstrap refits (`parallel::mclapply`).

</div>

<div class="section level2">

## Value

An object of class `saereg`: `global`, `features`, `joint` (tibbles),
`beta` (dimensions x terms), `weights` (C), `V_alpha` (per term), `null`
(the test's draws of the statistic, draws x terms), and the settings.

</div>

<div class="section level2">

## Details

**Overall effect** (`global`, one row per term):

-   `norm_sq_raw` = \|\|beta\|\|^2, biased upward by sampling noise;
    `norm_sq_cor` subtracts the noise, sum\_k V(beta\_k) (Green et al.
    2025): the HC2 variance without `cluster`, the cluster-jackknife
    variance with it.

-   `se`, `ci_lo`, `ci_hi`: jackknife standard error and normal interval
    of `norm_sq_cor`, with the correction held at its full-sample value.

-   `p`: global test of beta = 0. `test = "permutation"` (randomized
    designs, no clusters): Freedman-Lane, the residuals of the model
    without the term permuted, statistic \|\|beta\|\|^2.
    `test = "bootstrap"` (e.g. difference-in-differences): clusters (or
    documents) resampled within `strata`, statistic \|\|beta\_b -
    beta\|\|^2 against \|\|beta\|\|^2. Both report p = (1 +
    \#draws &gt;= observed) / (draws + 1).

**Decomposition** (`features`, one row per term x direction w\_j):

-   `alpha` = t(w\_j) beta: how far the texts move towards (&gt; 0) or
    away from (&lt; 0) w\_j.

-   `delta` = G^-1 alpha, G = t(W) W: the effect along w\_j with the
    other directions held fixed (beta regressed on all directions).

-   `q_j` = delta\_j alpha\_j / \|\|beta\|\|^2: direction j's share of
    the effect; the q\_j add up to R^2. `q_j_cor` subtracts the noise,
    (G^-1 V\_alpha)\_jj, and divides by `norm_sq_cor`.

-   Standard errors, intervals and normal p-values from the
    leave-one-out jackknife: closed form over documents without
    `cluster`, refits leaving out one cluster with it.

-   With `activations` (the SAE's a\_ij): `gamma`, the effect on the
    activation; `on_shift`, the effect on 1(a\_ij &gt; 0); `share_on`,
    the share of documents using j.

**Joint shares** (`joint`): R^2 = t(alpha) G^-1 alpha / \|\|beta\|\|^2,
the share of the effect in the span of all directions, raw and
corrected; the same for the two and three directions with the largest
`q_j_cor` (biased upward by that selection).

</div>

</div>
