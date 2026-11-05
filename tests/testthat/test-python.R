# Tests of the Python wrappers (2026-11-05), without API calls: a small SAE trained twice, the
# label prompts and the embedding plan on dry runs. Skipped unless the interpreter in
# getOption("saereg.python") / SAEREG_PYTHON has torch and hypothesaes.

py_ok <- function() {
  py <- path.expand(getOption("saereg.python", Sys.getenv("SAEREG_PYTHON", "python3")))
  out <- suppressWarnings(system2(py, c("-c", shQuote("import torch, hypothesaes, tiktoken")),
                                  stdout = FALSE, stderr = FALSE))
  identical(out, 0L)
}

test_that("train_sae: unit decoder directions, K active features, reproducible", {
  skip_if_not(py_ok(), "no Python with torch and hypothesaes")
  set.seed(1)
  E <- matrix(rnorm(300 * 16), 300, 16)
  E <- E / sqrt(rowSums(E^2))
  train <- rep(c(TRUE, FALSE), 150)
  a <- train_sae(E, M = 6, K = 2, seed = 3L, train = train)
  b <- train_sae(E, M = 6, K = 2, seed = 3L, train = train)
  expect_equal(dim(a$W), c(16L, 6L))
  expect_equal(unname(sqrt(colSums(a$W^2))), rep(1, 6), tolerance = 1e-6)
  expect_equal(dim(a$activations), c(300L, 6L))
  expect_true(all(a$activations >= 0) && all(rowSums(a$activations > 0) <= 2))
  expect_identical(a$W, b$W)
})

test_that("label_features and embed_texts: dry runs send nothing", {
  skip_if_not(py_ok(), "no Python with torch and hypothesaes")
  set.seed(2)
  E <- matrix(rnorm(60 * 8), 60, 8)
  W <- apply(matrix(rnorm(8 * 3), 8, 3), 2, function(w) w / sqrt(sum(w^2)))
  texts <- paste("text number", seq_len(60))
  p <- label_features(texts, E, W, dry_run = TRUE)
  expect_length(p, 3L)
  expect_true(all(grepl("GROUP 3 (score percentiles 90-100", p, fixed = TRUE)))
  expect_null(suppressMessages(embed_texts(c("one text", "another text"), dry_run = TRUE)))
})

test_that("a Python that does not exist stops with a clear message", {
  old <- options(saereg.python = "/no/such/python3")
  on.exit(options(old))
  expect_error(embed_texts("a text", dry_run = TRUE), "Python not found")
})
