# Tests of the Python wrappers (2026-11-05), without API calls: a small SAE trained twice, the
# label prompts and the embedding plan on dry runs. Skipped unless saereg_python() finds a
# Python with the required modules.

py_ok <- function() !is.na(saereg_python())

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

test_that("saereg_python() replaces a Python without the modules by one that has them", {
  skip_if_not(py_ok(), "no Python with torch and hypothesaes")
  old <- options(saereg.python = "/no/such/python3", saereg.python.checked = NULL)
  on.exit(options(old))
  expect_message(p <- saereg_python(), "not a Python with the required modules")
  expect_false(identical(p, "/no/such/python3"))
  expect_identical(getOption("saereg.python"), p)
})

test_that("active runs return the prompts and the model the API reports", {
  skip_if_not(py_ok(), "no Python with torch and hypothesaes")
  # A stand-in for the openai module: python -m puts the working directory first on its path,
  # so this file is imported instead of the real client and nothing reaches the API.
  stub <- tempfile("openai_stub_")
  dir.create(stub)
  writeLines(c(
    "import json",
    "class Obj:",
    "    def __init__(self, **kw): self.__dict__.update(kw)",
    "class Embeddings:",
    "    def create(self, input, model, **kw):",
    "        with open('calls.txt', 'a') as f: f.write(json.dumps(kw) + '\\n')",
    "        d = kw.get('dimensions', 4)",
    "        data = [Obj(embedding=[1.0] + [0.5] * (d - 1)) for _ in input]",
    "        return Obj(model=model + '-snapshot', data=data)",
    "class Completions:",
    "    def create(self, model, temperature, messages):",
    "        reader = messages[0]['content'].startswith('You will read')",
    "        text = 'YES' if reader else 'LABEL: \"a stub label\"'",
    "        msg = [Obj(message=Obj(content=text))]",
    "        return Obj(model=model + '-snapshot', choices=msg)",
    "class OpenAI:",
    "    def __init__(self, api_key):",
    "        self.embeddings = Embeddings()",
    "        self.chat = Obj(completions=Completions())"),
    file.path(stub, "openai.py"))
  wd <- setwd(stub)
  on.exit(setwd(wd), add = TRUE)
  key <- Sys.getenv("OPENAI_KEY_SAE", unset = NA)
  Sys.setenv(OPENAI_KEY_SAE = "stub")
  on.exit(if (is.na(key)) Sys.unsetenv("OPENAI_KEY_SAE") else Sys.setenv(OPENAI_KEY_SAE = key),
          add = TRUE)

  e4 <- embed_texts(c("one text", "another text"))
  expect_identical(dim(e4), c(2L, 4L))
  expect_identical(attr(e4, "model_used"), "text-embedding-3-large-snapshot")
  e8 <- embed_texts(c("one text", "another text"), dimensions = 8)
  expect_identical(ncol(e8), 8L)
  expect_identical(readLines("calls.txt"), c("{}", "{\"dimensions\": 8}"))

  set.seed(3)
  E <- matrix(rnorm(200 * 8), 200, 8)              # enough texts for unseen ones in each bin
  W <- apply(matrix(rnorm(8 * 3), 8, 3), 2, function(w) w / sqrt(sum(w^2)))
  texts <- paste("text number", seq_len(200))
  dry <- label_features(texts, E, W, dry_run = TRUE)
  lab <- label_features(texts, E, W)
  expect_identical(names(lab), c("feature", "label", "prompt", "model_used"))
  expect_identical(lab$prompt, dry)
  expect_identical(lab$model_used, rep("gpt-4.1-snapshot", 3L))
  fid <- label_features(texts, E, W, fidelity = TRUE)
  expect_identical(names(fid), c("feature", "label", "prompt", "model_used", "precision",
                                 "recall", "f1", "fidelity_model_used"))
  expect_identical(fid$fidelity_model_used, rep("gpt-4.1-snapshot", 3L))
})

test_that("train_sae: the settings are saved, and a version warning reaches R", {
  skip_if_not(py_ok(), "no Python with torch and hypothesaes")
  set.seed(4)
  E <- matrix(rnorm(100 * 8), 100, 8)
  a <- train_sae(E, M = 6, K = 2, seed = 3L)
  meta <- yaml::read_yaml(file.path(a$dir, "sae", "meta.json"))
  expect_identical(meta$training$n_epochs, 100L)
  expect_identical(meta$training$batch_size, 512L)
  expect_identical(meta$training$aux_k, 4L)
  real <- run_python
  local_mocked_bindings(run_python = function(args) {
    out <- real(args)
    if (length(args) < 3L || args[3] != "sae") return(out)
    c(out, "[warning] hypothesaes 9 is installed")
  })
  expect_warning(train_sae(E, M = 6, K = 2, seed = 3L), "hypothesaes 9 is installed")
})

test_that("label_features: a custom prompt is filled in, and checked before anything is sent", {
  E <- matrix(rnorm(200 * 8), 200, 8)
  W <- apply(matrix(rnorm(8 * 3), 8, 3), 2, function(w) w / sqrt(sum(w^2)))
  texts <- paste("text {n_bins} number", seq_len(200))         # a placeholder inside the texts
  expect_error(label_features(texts, E, W, prompt = "No texts here. LABEL:", dry_run = TRUE),
               "must contain \\{groups\\}")
  skip_if_not(py_ok(), "no Python with torch and hypothesaes")
  tpl <- 'Texts:\n{groups}\nGroup {n_bins}, {max_label_words} words, {other}.\nLABEL: "<label>"'
  p <- label_features(texts, E, W, prompt = tpl, dry_run = TRUE,
                      settings = list(max_label_words = 9L))
  expect_length(p, 3L)
  expect_true(all(startsWith(p, "Texts:\nGROUP 1 (score percentiles 0-10")))
  expect_true(all(grepl("Group 3, 9 words, {other}.", p, fixed = TRUE)))
  expect_true(all(grepl("- text {n_bins} number", p, fixed = TRUE)))
  expect_false(any(grepl("{groups}", p, fixed = TRUE)))
  expect_warning(label_features(texts, E, W, prompt = "{groups} Name it.", dry_run = TRUE),
                 "LABEL")
})
