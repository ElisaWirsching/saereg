<div id="main" class="col-md-9" role="main">

# Immigration and punishment: open-ended answers from a survey experiment

<div class="ref-description section level2">

Experiment 3 of Egami et al. (2022). Respondents on Amazon Mechanical
Turk (September 2017) read about a 28-year-old man convicted of entering
the United States illegally. In the treatment condition he had
previously served two prison sentences, one for a violent crime, and had
been deported; in the control condition he had never been imprisoned.
Respondents said whether he should be sent to prison and explained what
the US government should do and why. The answers are split into a
training and a test half as in Egami et al.; the SAE and its labels were
learned on the training half, the effects are estimated on the test
half.

</div>

<div class="section level2">

## Usage

<div class="sourceCode">

``` r
egami
```

</div>

</div>

<div class="section level2">

## Format

A list; the rows of `data`, `E` and `activations` are the same answers
in the same order:

-   data:

    Data frame, 1,034 rows: `doc_id`, `train` (`TRUE` for the 521
    answers of the training half), `treat` (1 = criminal history),
    `prison` (the closed-ended answer: "Yes", "No", "Don't know", or
    `NA`), `text` (the open-ended answer).

-   E:

    1,034 x 3,072 matrix: the answers' embeddings (OpenAI
    text-embedding-3-large), as returned by `embed_texts()`.

-   W:

    3,072 x 7 matrix: decoder directions of a sparse autoencoder (M = 7,
    K = 2, seed 7) trained on the training half with `train_sae()`.

-   labels:

    The seven directions' labels (gpt-4.1, `label_features()` on the
    training half).

-   fidelity:

    Data frame, 7 rows: `feature`, `precision`, `recall`, `f1` of each
    label on training answers the labeller did not see.

-   activations:

    1,034 x 7 matrix: the SAE's activations of the answers.

</div>

<div class="section level2">

## Source

Egami, Fong, Grimmer, Roberts and Stewart (2022), "How to make causal
inferences using texts", *Science Advances* 8. Replication data: Harvard
Dataverse, doi:10.7910/DVN/QHJN8V, CC0 1.0. Embeddings, SAE and labels:
Wirsching (2026). Built by `data-raw/egami.R`.

</div>

</div>
