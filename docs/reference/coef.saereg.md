<div id="main" class="col-md-9" role="main">

# Coefficients and intervals of an SAE regression

<div class="ref-description section level2">

`coef()` returns delta (the effect along each direction, the others held
fixed; default) or alpha (the simple projection); `confint()` their
jackknife intervals at any level.

</div>

<div class="section level2">

## Usage

<div class="sourceCode">

``` r
# S3 method for class 'saereg'
coef(object, type = c("delta", "alpha"), ...)

# S3 method for class 'saereg'
confint(
  object,
  parm = NULL,
  level = object$level,
  type = c("delta", "alpha"),
  ...
)
```

</div>

</div>

<div class="section level2">

## Arguments

-   object:

    An `saereg` fit.

-   type:

    `"delta"` or `"alpha"`.

-   ...:

    Unused.

-   parm:

    Directions (feature numbers, from 0) to return; all by default.

-   level:

    Confidence level.

</div>

</div>
