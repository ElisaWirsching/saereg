<div id="main" class="col-md-9" role="main">

# Embed texts with the OpenAI embeddings API

<div class="ref-description section level2">

Texts longer than the model's input limit are split into chunks and
their embeddings averaged with token weights; requests are batched
(Python stage `embed`).

</div>

<div class="section level2">

## Usage

<div class="sourceCode">

``` r
embed_texts(
  texts,
  model = "text-embedding-3-large",
  dry_run = FALSE,
  dir = tempfile("saereg_embed_")
)
```

</div>

</div>

<div class="section level2">

## Arguments

-   texts:

    Character vector.

-   model:

    OpenAI embedding model.

-   dry\_run:

    `TRUE` reports the number of tokens and requests without calling the
    API.

-   dir:

    Run directory for the inputs and outputs (a new temporary one by
    default).

</div>

<div class="section level2">

## Value

A matrix, one row per text (unit length), or `NULL` on a dry run.

</div>

</div>
