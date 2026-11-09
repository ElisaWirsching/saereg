# The Python stages (inst/python/saereg_ml) called from R with plain inputs. Each function
# writes its inputs into a fresh run directory, runs one stage of the Python command line
# (python -m saereg_ml <stage> --config <dir>/config.yaml) and reads the outputs back. The
# Python interpreter is the one saereg_python() finds. API calls (embeddings, labels, fidelity)
# need the key in OPENAI_KEY_SAE and cost money; every such function has `dry_run = TRUE`,
# which sends nothing.
#
# Matrices go from R to Python as raw little-endian doubles that Python turns into Parquet with
# its own writer (decided 2026-11-05): Parquet written by R's arrow 24 cannot be read by
# pyarrow 19 ("Repetition level histogram size mismatch"), and pyarrow 19 aborts at exit after
# reading R's Feather files. Parquet written by Python is read by R without problems.

#' Embed texts with the OpenAI embeddings API
#'
#' Texts longer than the model's input limit are split into chunks and their embeddings
#' averaged with token weights; requests are batched (Python stage `embed`).
#'
#' @param texts Character vector.
#' @param model OpenAI embedding model.
#' @param dimensions Optional length of the embeddings, for models that can shorten them. By
#'   default nothing is sent and the model returns its full size.
#' @param dry_run `TRUE` reports the number of tokens and requests without calling the API.
#' @param dir Run directory for the inputs and outputs (a new temporary one by default).
#' @return A matrix, one row per text (unit length), with the model the API reports in
#'   `attr(, "model_used")`; `NULL` on a dry run.
#' @export
embed_texts <- function(texts, model = "text-embedding-3-large", dimensions = NULL,
                        dry_run = FALSE, dir = tempfile("saereg_embed_")) {
  if (anyNA(texts)) stop("`texts` has missing values")
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  ids <- sprintf("d%07d", seq_along(texts))
  utils::write.csv(data.frame(id = ids, text = as.character(texts), all = "all"),
                   file.path(dir, "texts.csv"), row.names = FALSE)
  cfg <- list(documents = list(file = normalizePath(file.path(dir, "texts.csv")), id = "id",
                               text = "text", cells = list(all = list(all = "all"))),
              embeddings = list(model = model, dimensions = if (!is.null(dimensions))
                as.integer(dimensions)))
  out <- run_python(c("-m", "saereg_ml", "embed", "--config", write_run(dir, cfg),
                      if (dry_run) "--dry-run"))
  if (dry_run) {
    message(paste(out, collapse = "\n"))
    return(invisible(NULL))
  }
  emb <- arrow::read_parquet(file.path(dir, "embeddings.parquet"))
  stopifnot(identical(as.character(emb$doc_id), ids))
  E <- unname(as.matrix(emb[, setdiff(names(emb), "doc_id")]))
  attr(E, "model_used") <- unlist(yaml::read_yaml(file.path(dir, "manifest_embed.json"))$model_used)
  E
}

#' Train a sparse autoencoder on embeddings
#'
#' A TopK SAE trained on the rows marked `train` with the implementation in HypotheSAEs
#' (Movva et al. 2025; Python stage `sae`); the trained encoder then gives the activations of
#' every row. NumPy and torch are seeded with `seed` right before training, so a run on the
#' same device reproduces the SAE. The training settings are fixed: 100 epochs, batch size 512,
#' learning rate 5e-4, gradient clipping at 1, and an auxiliary loss for dead features (weight
#' 1/32, `min(2K, M)` features, after 256 steps without firing). They are saved in the run's
#' `sae/meta.json`. A warning is given if the installed `hypothesaes` is not version 2.0.0, the
#' version these settings were verified with.
#'
#' @references Movva, R., Peng, K., Garg, N., Kleinberg, J. and Pierson, E. (2025). Sparse
#'   autoencoders for hypothesis generation. *Proceedings of the 42nd International Conference
#'   on Machine Learning*. <https://github.com/rmovva/HypotheSAEs>
#'
#' @param E Embeddings (documents x dimensions).
#' @param M Number of features (decoder directions).
#' @param K Features active per document.
#' @param seed Seed of the training.
#' @param train Logical, the rows the SAE is trained on (e.g. the training half). The others
#'   are only encoded.
#' @param dir Run directory (a new temporary one by default); it keeps `sae/sae.pt`.
#' @return A list: `W` (dimensions x M, unit-length decoder directions), `activations`
#'   (documents x M), `dir`.
#' @export
train_sae <- function(E, M, K, seed = 7L, train = rep(TRUE, nrow(E)),
                      dir = tempfile("saereg_sae_")) {
  stopifnot(is.logical(train), length(train) == nrow(E), any(train))
  cfg <- list(sae = list(M = as.integer(M), K = as.integer(K), seed = as.integer(seed)))
  out <- run_python(c("-m", "saereg_ml", "sae", "--config", write_run(dir, cfg, E, train)))
  w <- grep("^\\[warning\\] ", out, value = TRUE)              # e.g. another hypothesaes version
  if (length(w)) warning(sub("^\\[warning\\] ", "", w), call. = FALSE)
  dec <- arrow::read_parquet(file.path(dir, "sae", "decoder.parquet"))
  act <- arrow::read_parquet(file.path(dir, "sae", "activations.parquet"))
  list(W = unname(as.matrix(dec[, setdiff(names(dec), "dim")])),
       activations = unname(as.matrix(act[, setdiff(names(act), "doc_id")])), dir = dir)
}

#' Label SAE directions with a large language model
#'
#' For each direction w_j, the texts are ranked by their score e_i'w_j; the model sees
#' `n_per_bin` random texts from the bottom 10%, the middle 45-55% and the top 10% and names
#' what the top texts do that the others do not (Python stage `label`). With `fidelity = TRUE`,
#' a second call per text checks whether each label picks out its direction's top texts among
#' texts the labeller did not see (stage `fidelity`: precision, recall, F1). Pass the training
#' texts: the labels must not use the texts on which the effects are estimated.
#'
#' @param texts Character vector (the training texts).
#' @param E Their embeddings (documents x dimensions).
#' @param W Decoder directions (dimensions x M).
#' @param fidelity Also run the fidelity test (100 API calls per direction by default)?
#' @param dry_run `TRUE` returns the prompts without calling the API.
#' @param model,seed Labelling model and seed of the text sampling (direction j: seed + j).
#' @param settings Optional list overriding the defaults: `bins`, `n_per_bin` (7),
#'   `max_words` (256), `max_label_words` (15), `temperature` (0), `n_top` and `n_lower` (50
#'   each, fidelity).
#' @param dir Run directory (a new temporary one by default).
#' @return A tibble with `feature` (from 0), `label`, `prompt` (the prompt sent to the model)
#'   and `model_used` (the model version the API reports), plus `precision`, `recall`, `f1` and
#'   `fidelity_model_used` with `fidelity = TRUE`; on a dry run, the prompts (character vector).
#' @export
label_features <- function(texts, E, W, fidelity = FALSE, dry_run = FALSE, model = "gpt-4.1",
                           seed = 7L, settings = list(), dir = tempfile("saereg_label_")) {
  stopifnot(length(texts) == nrow(E), nrow(W) == ncol(E), !anyNA(texts))
  st <- utils::modifyList(list(bins = list(c(0L, 10L), c(45L, 55L), c(90L, 100L)),
                               n_per_bin = 7L, max_words = 256L, max_label_words = 15L,
                               temperature = 0, n_top = 50L, n_lower = 50L), settings)
  cfg <- list(labels = list(model = model, temperature = st$temperature, bins = st$bins,
                            n_per_bin = st$n_per_bin, max_words = st$max_words,
                            max_label_words = st$max_label_words, seed = as.integer(seed)),
              fidelity = list(model = model, temperature = st$temperature, n_top = st$n_top,
                              n_lower = st$n_lower, max_words = st$max_words,
                              seed = as.integer(seed)))
  cfg_file <- write_run(dir, cfg, E, rep(TRUE, nrow(E)), texts, W)
  # The prompts come from a dry run in both modes: the stage builds them with the same seeded
  # draw whether it writes them out or sends them, so these are the prompts the model receives.
  run_python(c("-m", "saereg_ml", "label", "--config", cfg_file, "--dry-run"))
  pd <- file.path(dir, "labels", "prompts_dryrun")
  prompts <- vapply(seq_len(ncol(W)) - 1L, function(j) {       # the files as written
    f <- file.path(pd, sprintf("feature_%d.txt", j))
    x <- readChar(f, file.info(f)$size, useBytes = TRUE)
    Encoding(x) <- "UTF-8"
    x
  }, character(1))
  if (dry_run) return(prompts)
  run_python(c("-m", "saereg_ml", "label", "--config", cfg_file))
  out <- tibble::as_tibble(utils::read.csv(file.path(dir, "labels", "labels.csv")))
  stopifnot(identical(out$feature, seq_len(ncol(W)) - 1L))
  out$prompt <- prompts
  out$model_used <- unlist(yaml::read_yaml(file.path(dir, "manifest_label.json"))$model_used)
  if (fidelity) {
    run_python(c("-m", "saereg_ml", "fidelity", "--config", cfg_file))
    fid <- utils::read.csv(file.path(dir, "labels", "label_fidelity.csv"))
    out <- tibble::as_tibble(merge(out, fid[c("feature", "precision", "recall", "f1")],
                                   by = "feature", sort = TRUE))
    out$fidelity_model_used <- paste(unlist(yaml::read_yaml(
      file.path(dir, "manifest_fidelity.json"))$model_used), collapse = ", ")
  }
  out
}

# Writes what the Python stages read into `dir` and returns the config's path: config.yaml
# (with run_dir) and, given E, documents.csv, split.csv (is_train = train), embeddings.parquet
# and, given W, sae/decoder.parquet. The matrices pass as raw doubles that Python writes as
# Parquet (see the top of this file).
write_run <- function(dir, cfg, E = NULL, train = NULL, texts = NULL, W = NULL) {
  dir.create(file.path(dir, "sae"), recursive = TRUE, showWarnings = FALSE)
  dir <- normalizePath(dir)
  cfg$run_dir <- dir
  yaml::write_yaml(cfg, file.path(dir, "config.yaml"))
  if (!is.null(E)) {
    ids <- sprintf("d%07d", seq_len(nrow(E)))
    txt <- if (is.null(texts)) "" else as.character(texts)
    utils::write.csv(data.frame(doc_id = ids, cell = "all", text = txt),
                     file.path(dir, "documents.csv"), row.names = FALSE)
    utils::write.csv(data.frame(doc_id = ids, cell = "all", is_train = as.integer(train)),
                     file.path(dir, "split.csv"), row.names = FALSE)
    mats <- list(E = E, W = W)
    mats <- mats[!vapply(mats, is.null, logical(1))]
    for (nm in names(mats)) {                                  # row-major, little-endian
      con <- file(file.path(dir, paste0(nm, ".f64")), "wb")
      writeBin(as.vector(t(mats[[nm]])), con, size = 8L, endian = "little")
      close(con)
    }
    py <- c("import os", "import numpy as np", "from saereg_ml.io import write_matrix",
            sprintf("d = %s", deparse(dir)),
            sprintf("E = np.fromfile(os.path.join(d, 'E.f64'), dtype='<f8').reshape(%d, %d)",
                    nrow(E), ncol(E)),
            sprintf("ids = ['d%%07d' %% i for i in range(1, %d)]", nrow(E) + 1L),
            "write_matrix(os.path.join(d, 'embeddings.parquet'), ids, E, 'doc_id', 'e')")
    if (!is.null(W)) {
      py <- c(py,
              sprintf("W = np.fromfile(os.path.join(d, 'W.f64'), dtype='<f8').reshape(%d, %d)",
                      nrow(W), ncol(W)),
              sprintf("write_matrix(os.path.join(d, 'sae', 'decoder.parquet'), np.arange(%d), W,",
                      nrow(W)),
              "             'dim', 'f')")
    }
    writeLines(py, file.path(dir, "write_matrices.py"))
    run_python(file.path(dir, "write_matrices.py"))
    unlink(file.path(dir, c("E.f64", "W.f64", "write_matrices.py")))
  }
  file.path(dir, "config.yaml")
}

#' The Python that saereg uses
#'
#' Returns the first Python that has the modules the Python stages need (numpy, pandas,
#' pyarrow, pyyaml, torch, hypothesaes, tiktoken, openai), trying in turn
#' `getOption("saereg.python")`, the environment variable `SAEREG_PYTHON`, `python3`, and the
#' usual Anaconda, Miniconda, Miniforge and Homebrew locations. The result is kept in
#' `options(saereg.python)` for the session. Set that option (or `SAEREG_PYTHON`, e.g. in
#' `~/.Renviron`) to choose a Python yourself.
#'
#' @return The path of the Python, or `NA` if none has the modules.
#' @export
saereg_python <- function() {
  set <- getOption("saereg.python", "")
  if (nzchar(set) && identical(set, getOption("saereg.python.checked"))) return(set)
  wanted <- c(set, Sys.getenv("SAEREG_PYTHON", ""))
  wanted <- path.expand(wanted[nzchar(wanted)])
  cand <- unique(c(wanted, "python3", path.expand(c(
    "~/anaconda3/bin/python3", "~/miniconda3/bin/python3", "~/miniforge3/bin/python3",
    "~/mambaforge/bin/python3", "/opt/anaconda3/bin/python3", "/opt/miniconda3/bin/python3",
    "/opt/homebrew/bin/python3", "/usr/local/bin/python3"))))
  imports <- "import numpy, pandas, pyarrow, yaml, torch, hypothesaes, tiktoken, openai"
  for (p in cand) {
    ok <- nzchar(Sys.which(p)) && identical(suppressWarnings(system2(
      p, c("-c", shQuote(imports)), stdout = FALSE, stderr = FALSE)), 0L)
    if (ok) {
      if (length(wanted) && !p %in% wanted) {
        message("saereg: '", wanted[1], "' is not a Python with the required modules; ",
                "using '", p, "'.")
      }
      options(saereg.python = p, saereg.python.checked = p)
      return(p)
    }
  }
  NA_character_
}

# Runs the Python interpreter with the package's Python module on the path; stops with the
# last lines of output if it fails. Returns the printed output.
run_python <- function(args) {
  py <- saereg_python()
  if (is.na(py)) {
    stop("No Python with numpy, pandas, pyarrow, pyyaml, torch, hypothesaes, tiktoken and ",
         "openai found. Install them, or point saereg to such a Python: ",
         "options(saereg.python = \"/path/to/python3\") or the environment variable ",
         "SAEREG_PYTHON.", call. = FALSE)
  }
  pypath <- system.file("python", package = "saereg")
  out <- suppressWarnings(system2(py, args, env = paste0("PYTHONPATH=", shQuote(pypath)),
                                  stdout = TRUE, stderr = TRUE))
  status <- attr(out, "status")
  if (!is.null(status) && status != 0) {
    stop("Python failed (", py, " ", args[1], " ", args[2], "):\n",
         paste(utils::tail(out, 15), collapse = "\n"))
  }
  out
}
