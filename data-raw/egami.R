# Producer of the package data `egami` (data/egami.rda), for the getting-started vignette.
#
# Inputs (project files, not shipped):
#   - Code/applications/egami_immigration/outputs/run_large_M7_K2/: documents.csv, split.csv,
#     embeddings.parquet, sae/decoder.parquet, sae/activations.parquet, labels/labels.csv,
#     labels/label_fidelity.csv (the paper's Egami run: text-embedding-3-large; SAE M = 7,
#     K = 2, seed 7, trained on the training half; labels and fidelity by gpt-4.1)
#   - Data/egami_immigration/egami_exp3_clean.csv (treat, the closed-ended prison answer)
# Output: data/egami.rda, a list with
#   data (1,034 answers: doc_id, train, treat, prison, text), E (1,034 x 3,072 embeddings),
#   W (3,072 x 7 decoder directions), labels (7), fidelity (7 rows: feature, precision,
#   recall, f1), activations (1,034 x 7).
# Decisions:
#   - Both halves (EW, 2026-11-05; first the test half only, the same day): the vignette walks
#     through the whole pipeline, so readers can re-train the SAE on the training half and
#     estimate on the test half. About 6 MB.
#   - Embeddings at full double precision (xz-compressed as well as integers scaled by 1e7).
#   - Licence: Egami et al. (2022) replication data, CC0 1.0 (Data/egami_immigration/README.md).
#   - prison: "" in the clean file becomes NA (respondents who skipped the question).
#   - The 60 empty answers are not included (the run left them out, as Egami et al.).
# Run from the project root: Rscript Code/pipeline/R/saereg/data-raw/egami.R

library(dplyr)
library(tidylog)

root    <- file.path(path.expand("~"), "Dropbox", "Research", "EmbeddingSAE")
run_dir <- file.path(root, "Code", "applications", "egami_immigration", "outputs",
                     "run_large_M7_K2")
pkg_dir <- file.path(root, "Code", "pipeline", "R", "saereg")

# The run's files, all in the same document order (read.csv: readr would trim the texts)
docs  <- read.csv(file.path(run_dir, "documents.csv"), colClasses = "character")
split <- read.csv(file.path(run_dir, "split.csv"),
                  colClasses = c("character", "character", "integer"))
emb   <- arrow::read_parquet(file.path(run_dir, "embeddings.parquet"))
act   <- arrow::read_parquet(file.path(run_dir, "sae", "activations.parquet"))
dec   <- arrow::read_parquet(file.path(run_dir, "sae", "decoder.parquet"))
lab   <- read.csv(file.path(run_dir, "labels", "labels.csv"))
fid   <- read.csv(file.path(run_dir, "labels", "label_fidelity.csv"))
stopifnot(identical(split$doc_id, docs$doc_id), identical(as.character(emb$doc_id), docs$doc_id),
          identical(as.character(act$doc_id), docs$doc_id), !anyDuplicated(docs$doc_id),
          identical(lab$feature, 0:6), identical(fid$feature, 0:6), identical(fid$label, lab$label),
          identical(as.numeric(dec$dim), as.numeric(0:3071)))

# Treatment and prison answer from the clean file; doc_id is its row_id (unique)
clean <- read.csv(file.path(root, "Data", "egami_immigration", "egami_exp3_clean.csv"),
                  colClasses = c(row_id = "character"))
stopifnot(!anyDuplicated(clean$row_id))
d <- tibble(doc_id = docs$doc_id, cell = docs$cell, train = split$is_train == 1L,
            text = docs$text) %>%
  left_join(clean %>% select(row_id, treat, prison, text_clean = text),
            by = c("doc_id" = "row_id"))
# every answer of the run is in the clean file (the run was built from it); the unmatched
# clean rows are the 60 empty answers, which the run left out
stopifnot(nrow(d) == 1034L, !anyNA(d$treat), identical(d$text, d$text_clean),
          identical(d$treat, as.integer(d$cell == "treated")))
unmatched <- anti_join(clean, d, by = c("row_id" = "doc_id"))
stopifnot(nrow(unmatched) == 60L, all(unmatched$text_empty))

egami <- list(
  data = d %>% transmute(doc_id = as.integer(doc_id), train, treat, prison = na_if(prison, ""),
                         text),
  E = unname(as.matrix(emb[, -1])),
  W = unname(as.matrix(dec[, -1])),
  labels = lab$label,
  fidelity = fid %>% select(feature, precision, recall, f1),
  activations = unname(as.matrix(act[, -1])))
stopifnot(nrow(egami$E) == 1034L, ncol(egami$E) == 3072L, dim(egami$W) == c(3072L, 7L),
          sum(egami$data$train) == 521L, all(abs(sqrt(colSums(egami$W^2)) - 1) < 1e-5))

dir.create(file.path(pkg_dir, "data"), showWarnings = FALSE)
out <- file.path(pkg_dir, "data", "egami.rda")
save(egami, file = out, compress = "xz")
chk <- new.env(); load(out, envir = chk)
stopifnot(identical(chk$egami, egami))                       # round trip
cat(sprintf("written %s (%.1f MB): %d answers (training half %d; treated %d), %d dimensions, %d directions\n",
            out, file.size(out) / 1e6, nrow(egami$data), sum(egami$data$train),
            sum(egami$data$treat), ncol(egami$E), ncol(egami$W)))
