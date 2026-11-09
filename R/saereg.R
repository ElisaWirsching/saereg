#' SAE regression: an embedding regression decomposed onto SAE directions
#'
#' Regresses the document embeddings on a design (OLS, covariates and absorbed fixed effects
#' allowed), takes the coefficient vector of the term of interest, beta, as the effect on the
#' texts, tests it globally, and decomposes it onto the decoder directions W of a sparse
#' autoencoder trained on other texts. Every estimate is a linear combination of the documents,
#' computed from the design alone (Frisch-Waugh-Lovell).
#'
#' **Overall effect** (`global`, one row per term):
#' * `norm_sq_raw` = ||beta||^2, biased upward by sampling noise; `norm_sq_cor` subtracts the
#'   noise, sum_k V(beta_k) (Green et al. 2025): the HC2 variance without `cluster`, the
#'   cluster-jackknife variance with it.
#' * `se`, `ci_lo`, `ci_hi`: jackknife standard error and normal interval of `norm_sq_cor`,
#'   with the correction held at its full-sample value.
#' * `p`: global test of beta = 0. `test = "permutation"` (randomized designs, no clusters):
#'   Freedman-Lane, the residuals of the model without the term permuted, statistic
#'   ||beta||^2. `test = "bootstrap"` (e.g. difference-in-differences): clusters (or
#'   documents) resampled within `strata`, statistic ||beta_b - beta||^2 against ||beta||^2.
#'   Both report p = (1 + #{draws >= observed}) / (draws + 1).
#'
#' **Decomposition** (`features`, one row per term x direction w_j):
#' * `alpha` = t(w_j) beta: how far the texts move towards (> 0) or away from (< 0) w_j.
#' * `delta` = G^-1 alpha, G = t(W) W: the effect along w_j with the other directions held
#'   fixed (beta regressed on all directions).
#' * `q_j` = delta_j alpha_j / ||beta||^2: direction j's share of the effect; the q_j add up
#'   to R^2. `q_j_cor` subtracts the noise, (G^-1 V_alpha)_jj, and divides by `norm_sq_cor`.
#' * Standard errors, intervals and normal p-values from the leave-one-out jackknife: closed
#'   form over documents without `cluster`, refits leaving out one cluster with it.
#' * With `activations` (the SAE's a_ij): `gamma`, the effect on the activation;
#'   `on_shift`, the effect on 1(a_ij > 0); `share_on`, the share of documents using j.
#'
#' **Joint shares** (`joint`): R^2 = t(alpha) G^-1 alpha / ||beta||^2, the share of the effect
#' in the span of all directions, raw and corrected; the same for the two and three
#' directions with the largest `q_j_cor` (biased upward by that selection).
#'
#' **Scores** (`s`, `s_tilde`; one row per row of `data`, `NA` for rows dropped for missing
#' values, one column per direction): `s` holds s_ij = t(e_i) w_j, how far text i reaches along
#' w_j; `s_tilde` holds its coordinates G^-1 s_i, with the overlap between directions removed.
#' Regressing them on the design gives `alpha` and `delta`.
#'
#' @param formula `E ~ treat + covariates | fe1 + fe2`. The left side is the embedding matrix
#'   (documents x dimensions), found in `data` or in the formula's environment. Fixed effects
#'   after `|` are absorbed (`fixest::demean()`); combine variables beforehand for
#'   interacted effects (e.g. member x stack).
#' @param data Data frame with the variables. Rows with a missing value are dropped.
#' @param W Decoder directions (dimensions x M), unit-length columns, from an SAE trained on
#'   texts not used here.
#' @param term Column(s) of the design whose effect is decomposed; default the first variable
#'   on the right side. Several terms (e.g. party-specific treatment indicators) are
#'   estimated jointly and decomposed one by one.
#' @param cluster One-sided formula, e.g. `~ member`: jackknife and bootstrap by cluster.
#'   Required with fixed effects or `rows`.
#' @param strata One-sided formula, constant within clusters: the bootstrap resamples within
#'   each stratum (e.g. ever-treated and never-treated members).
#' @param test `"permutation"` or `"bootstrap"`.
#' @param draws Permutations or bootstrap draws.
#' @param seed Permutations: `set.seed(seed)` once, then one draw after another. Bootstrap:
#'   draw b uses `set.seed(seed + b)`, so any draw can be re-run on its own.
#' @param level Confidence level of the intervals.
#' @param activations Optional SAE activations (documents x M, rows as `E`).
#' @param labels Optional labels of the M directions.
#' @param rows Optional row of `E` for each row of `data`, when rows share an embedding (e.g.
#'   a member-day's mean in several stacks).
#' @param cores Cores for the jackknife and bootstrap refits (`parallel::mclapply`).
#' @return An object of class `saereg`: `global`, `features`, `joint` (tibbles), `beta`
#'   (dimensions x terms), `s` and `s_tilde` (rows of `data` x directions), `V_alpha` (per
#'   term), `null` (the test's draws of the statistic, draws x terms), and the settings.
#' @export
saereg <- function(formula, data, W, term = NULL, cluster = NULL, strata = NULL,
                   test = c("permutation", "bootstrap"), draws = 999L, seed = 7L,
                   level = 0.95, activations = NULL, labels = NULL, rows = NULL,
                   cores = 1L) {
  call <- match.call()
  test <- match.arg(test)
  env  <- environment(formula)

  # ---- 1. Design: E ~ covariates | fixed effects --------------------------------------
  rhs    <- formula[[3]]
  has_fe <- is.call(rhs) && identical(rhs[[1]], as.name("|"))
  f_x    <- stats::as.formula(call("~", if (has_fe) rhs[[2]] else rhs), env = env)
  fe_v   <- if (has_fe) all.vars(rhs[[3]]) else character(0)
  cl_v   <- if (is.null(cluster)) character(0) else all.vars(cluster)
  st_v   <- if (is.null(strata)) character(0) else all.vars(strata)
  if (has_fe && is.null(cluster)) {
    stop("fixed effects need `cluster`: the document jackknife needs leverages, which the ",
         "absorbed fixed effects do not give")
  }
  if (!is.null(rows) && is.null(cluster)) stop("`rows` needs `cluster`")
  if (test == "permutation" && !is.null(cluster)) {
    stop("the permutation test assumes independent documents; use test = \"bootstrap\"")
  }
  E <- as.matrix(eval(formula[[2]], data, env))
  if (!is.numeric(E)) stop("the left side of the formula must be a numeric matrix")
  if (is.null(rows) && nrow(E) != nrow(data)) stop("E and data have different numbers of rows")
  if (nrow(W) != ncol(E)) stop("W has ", nrow(W), " rows, E has ", ncol(E), " columns")
  if (any(abs(sqrt(colSums(W^2)) - 1) > 1e-5)) stop("the columns of W must have unit length")
  M <- ncol(W)
  if (!is.null(labels) && length(labels) != M) stop("`labels` needs one label per column of W")

  keep <- stats::complete.cases(data[unique(c(all.vars(f_x), fe_v, cl_v, st_v))])
  d <- data[keep, , drop = FALSE]
  if (is.null(rows)) {                       # E (and activations) follow the rows of data
    E <- E[keep, , drop = FALSE]
    if (!is.null(activations)) activations <- as.matrix(activations)[keep, , drop = FALSE]
  } else {
    rows <- as.integer(rows)[keep]
  }
  X <- stats::model.matrix(f_x, d)
  if (has_fe) X <- X[, colnames(X) != "(Intercept)", drop = FALSE]   # absorbed
  if (is.null(term)) term <- setdiff(colnames(X), "(Intercept)")[1]
  if (!all(term %in% colnames(X))) {
    stop("term not in the design: ", paste(colnames(X), collapse = ", "))
  }
  fe <- if (has_fe) lapply(d[fe_v], function(v) as.integer(factor(v))) else NULL
  n  <- nrow(X)
  nt <- length(term)
  z  <- stats::qnorm(1 - (1 - level) / 2)
  A  <- activations
  A_on <- if (is.null(A)) NULL else (A > 0) * 1

  # Estimates from the rows ii (with repeats in the bootstrap) and fixed-effect ids fe_ii:
  # beta (D x terms) and, with activations, gamma and on_shift (M x terms). Used for the
  # point estimate and for every jackknife and bootstrap refit, so all share one route.
  # The weights are added up per row of E (rows used twice, by `rows` or by the bootstrap, add
  # their weights) and multiplied with the whole of E, which is never copied.
  fit_rows <- function(ii, fe_ii) {
    C  <- fwl_weights(X[ii, , drop = FALSE], term, fe_ii)
    s  <- rowsum(C, if (is.null(rows)) ii else rows[ii])
    Cf <- matrix(0, nrow(E), ncol(C))
    Cf[as.integer(rownames(s)), ] <- s
    list(C = C, beta = crossprod(E, Cf),
         gamma = if (is.null(A)) NULL else crossprod(A, Cf),
         on = if (is.null(A)) NULL else crossprod(A_on, Cf))
  }
  pt   <- fit_rows(seq_len(n), fe)
  beta <- pt$beta
  colnames(beta) <- term
  norm_raw <- unname(colSums(beta^2))
  Ginv <- solve(crossprod(W))
  alpha <- crossprod(W, beta)                                  # M x terms
  delta <- Ginv %*% alpha

  # ---- 2. Variances ---------------------------------------------------------------------
  # Per term, the leave-one-out replicates of every reported quantity, one column each (the
  # norm, alpha, delta, then gamma and on_shift), give all jackknife SEs at once (section 4),
  # with factor f = (n - 1) / n. Without `cluster` the replicates are in closed form,
  # beta_(-i) = beta - g_i r_i; the noise in ||beta||^2 (tr_V) and the covariance of alpha
  # (V_alpha) are HC2. With `cluster`, each replicate refits the weights without one cluster,
  # and tr_V and V_alpha come from the replicates (as 06_did_results.R).
  inf <- vector("list", nt)
  if (is.null(cluster)) {
    C    <- pt$C
    XtXi <- solve(crossprod(X))
    h    <- rowSums((X %*% XtXi) * X)                          # leverages
    if (any(h > 1 - 1e-8)) stop("a document has leverage 1 (e.g. alone in its cell)")
    RE <- E - X %*% (XtXi %*% crossprod(X, E))                 # residual embeddings
    RS <- RE %*% W                                             # residual scores
    RA <- if (is.null(A)) NULL else A - X %*% (XtXi %*% crossprod(X, A))
    RO <- if (is.null(A)) NULL else A_on - X %*% (XtXi %*% crossprod(X, A_on))
    f  <- (n - 1) / n
    for (k in seq_len(nt)) {
      cw <- C[, k]
      g  <- cw / (1 - h)
      # replicates minus the estimate (constants do not change a variance)
      JK <- cbind(-2 * g * drop(RE %*% beta[, k]) + g^2 * rowSums(RE^2),   # ||beta_(-i)||^2
                  -g * RS, -g * (RS %*% Ginv),
                  if (is.null(A)) NULL else cbind(-g * RA, -g * RO))
      inf[[k]] <- list(JK = JK, tr_V = sum(colSums(cw^2 / (1 - h) * RE^2)),
                       V_alpha = crossprod(RS * sqrt(cw^2 / (1 - h))))
    }
  } else {
    cl_id <- interaction(d[cl_v], drop = TRUE, lex.order = TRUE)
    units <- levels(cl_id)
    G <- length(units)
    J <- do.call(rbind, parallel::mclapply(units, function(u) {
      ii <- which(cl_id != u)
      r  <- fit_rows(ii, if (has_fe) lapply(fe, `[`, ii) else NULL)
      c(r$beta, r$gamma, r$on)                                 # one replicate as a row
    }, mc.cores = cores))
    f <- (G - 1) / G
    D <- ncol(E)
    for (k in seq_len(nt)) {
      Jb <- J[, (k - 1) * D + seq_len(D), drop = FALSE]        # replicates of beta_k
      AJ <- Jb %*% W
      ga <- nt * D + (k - 1) * M + seq_len(M)                  # columns of gamma_k in J
      JK <- cbind(rowSums(Jb^2), AJ, AJ %*% Ginv,
                  if (is.null(A)) NULL else J[, c(ga, ga + nt * M), drop = FALSE])
      inf[[k]] <- list(JK = JK, tr_V = f * sum(sweep(Jb, 2, colMeans(Jb))^2),
                       V_alpha = f * crossprod(sweep(AJ, 2, colMeans(AJ))))
    }
  }

  # ---- 3. Global test ------------------------------------------------------------------
  if (test == "permutation") {
    # Freedman-Lane: permuted weights applied to the residuals of the model without the
    # term. The weights are orthogonal to the other columns, so the observed statistic is
    # ||beta||^2. One set.seed, draws in sequence: the draws of saereg 0.1.0.
    stat <- vapply(seq_len(nt), function(k) {
      X0 <- X[, colnames(X) != term[k], drop = FALSE]
      R0 <- if (ncol(X0) > 0) qr.resid(qr(X0), E) else E
      set.seed(seed)
      Cp <- vapply(seq_len(draws), function(b) pt$C[sample.int(n), k], numeric(n))
      colSums(crossprod(R0, Cp)^2)
    }, numeric(draws))
  } else {
    # Bootstrap: clusters (documents without `cluster`) drawn with replacement within each
    # stratum, strata and clusters in order of first appearance. A cluster drawn twice gets
    # its own ids for the fixed effects nested in clusters (e.g. member x stack).
    unit_id <- if (is.null(cluster)) factor(seq_len(n)) else cl_id
    st_id   <- if (is.null(strata)) rep("all", n) else
      as.character(interaction(d[st_v], drop = TRUE))
    if (any(tapply(st_id, unit_id, function(s) length(unique(s))) > 1)) {
      stop("`strata` must be constant within clusters")
    }
    first   <- !duplicated(unit_id)
    by_st   <- split(as.character(unit_id[first]), factor(st_id[first], unique(st_id[first])))
    rows_of <- split(seq_len(n), unit_id)
    nested  <- if (has_fe) vapply(fe, function(f) all(tapply(as.integer(unit_id), f,
                                    function(u) length(unique(u))) == 1), logical(1)) else NULL
    stat <- do.call(rbind, parallel::mclapply(seq_len(draws), function(b) {
      set.seed(seed + b)
      drawn <- unlist(lapply(by_st, function(u) u[sample.int(length(u), length(u),
                                                             replace = TRUE)]))
      ii   <- rows_of[drawn]
      copy <- rep(seq_along(ii), lengths(ii))
      ii   <- unlist(ii, use.names = FALSE)
      fe_b <- if (has_fe) Map(function(f, nst) if (nst) (copy - 1L) * max(f) + f[ii] else f[ii],
                              fe, nested) else NULL
      colSums((fit_rows(ii, fe_b)$beta - beta)^2)
    }, mc.cores = cores))
  }
  stat <- matrix(stat, draws, nt, dimnames = list(NULL, term))   # null draws x terms
  p_global <- unname((1 + colSums(sweep(stat, 2, norm_raw, ">="))) / (draws + 1))

  # ---- 4. Tables -----------------------------------------------------------------------
  glob <- feat <- jnt <- vector("list", nt)
  V_alpha <- stats::setNames(vector("list", nt), term)
  for (k in seq_len(nt)) {
    a <- alpha[, k]; dl <- delta[, k]; V <- inf[[k]]$V_alpha
    JK <- inf[[k]]$JK
    se <- sqrt(f * colSums(sweep(JK, 2, colMeans(JK))^2))    # norm, alpha, delta, gamma, on
    se_a <- se[1 + seq_len(M)]
    se_d <- se[1 + M + seq_len(M)]
    nc <- norm_raw[k] - inf[[k]]$tr_V
    glob[[k]] <- tibble::tibble(term = term[k], norm_sq_raw = norm_raw[k], norm_sq_cor = nc,
                                se = se[1], ci_lo = nc - z * se[1], ci_hi = nc + z * se[1],
                                p = p_global[k])
    q_cor <- (dl * a - diag(Ginv %*% V)) / nc
    ft <- tibble::tibble(
      term = term[k], feature = seq_len(M) - 1L,
      label = if (is.null(labels)) NA_character_ else as.character(labels),
      alpha = a, alpha_se = se_a, alpha_lo = a - z * se_a, alpha_hi = a + z * se_a,
      alpha_p = 2 * stats::pnorm(-abs(a / se_a)),
      delta = dl, delta_se = se_d, delta_lo = dl - z * se_d, delta_hi = dl + z * se_d,
      delta_p = 2 * stats::pnorm(-abs(dl / se_d)),
      q_j = dl * a / norm_raw[k], q_j_cor = q_cor)
    if (!is.null(A)) {
      se_g <- se[1 + 2 * M + seq_len(M)]
      se_o <- se[1 + 3 * M + seq_len(M)]
      gm <- pt$gamma[, k]; on <- pt$on[, k]
      ft$gamma <- gm; ft$gamma_se <- se_g
      ft$gamma_lo <- gm - z * se_g; ft$gamma_hi <- gm + z * se_g
      ft$on_shift <- on; ft$on_se <- se_o
      ft$on_lo <- on - z * se_o; ft$on_hi <- on + z * se_o
      ft$share_on <- colMeans((if (is.null(rows)) A else A[unique(rows), , drop = FALSE]) > 0)
    }
    feat[[k]] <- ft
    # joint shares of all directions and of the top 2 and top 3 by q_j_cor, raw and corrected:
    # t(alpha_S) G_S^-1 alpha_S / ||beta||^2, minus tr(G_S^-1 V_S) when corrected
    top <- order(-q_cor)
    js <- vapply(list(seq_len(M), top[seq_len(min(2, M))], top[seq_len(min(3, M))]), function(S) {
      GS  <- crossprod(W[, S, drop = FALSE])
      num <- drop(crossprod(a[S], solve(GS, a[S])))
      c(num / norm_raw[k], (num - sum(diag(solve(GS, V[S, S, drop = FALSE])))) / nc)
    }, numeric(2))
    jnt[[k]] <- tibble::tibble(term = term[k], R2 = js[1, 1], R2_cor = js[2, 1],
                               top2 = js[1, 2], top2_cor = js[2, 2],
                               top3 = js[1, 3], top3_cor = js[2, 3])
    V_alpha[[k]] <- V
  }

  # ---- 5. Scores -----------------------------------------------------------------------
  # Per row of `data`, the outcomes whose regressions on the design give alpha and delta. Rows
  # dropped above stay in as NA, so the matrices line up with `data` (added 2026-11-09, EW).
  # Multiplied before indexing, because E[rows, ] would copy all of E.
  S <- E %*% W
  if (!is.null(rows)) S <- S[rows, , drop = FALSE]
  s <- s_tilde <- matrix(NA_real_, nrow(data), M,
                         dimnames = list(NULL, paste0("f", seq_len(M) - 1L)))
  s[keep, ] <- S
  s_tilde[keep, ] <- S %*% Ginv

  structure(list(call = call, formula = formula, term = term, n = n,
                 n_clusters = if (is.null(cluster)) NA_integer_ else G,
                 n_dropped = sum(!keep), test = test, draws = draws, seed = seed,
                 level = level, global = do.call(rbind, glob),
                 features = do.call(rbind, feat), joint = do.call(rbind, jnt),
                 beta = beta, s = s, s_tilde = s_tilde, V_alpha = V_alpha, null = stat),
            class = "saereg")
}

# Row weights of the term(s), C = R (R'R)^-1 with R the term columns residualized on the
# fixed effects (fixest::demean) and on the other columns, so that beta = t(E) C is the OLS
# coefficient (Frisch-Waugh-Lovell). Without fixed effects C = X (X'X)^-1 [, term]. The only
# helper: every jackknife and bootstrap refit recomputes the weights.
fwl_weights <- function(X, term, fe = NULL) {
  if (!is.null(fe)) {
    X[] <- fixest::demean(X, f = fe, tol = 1e-10, iter = 100000L, notes = FALSE)
  }
  Xt <- X[, term, drop = FALSE]
  Xo <- X[, setdiff(colnames(X), term), drop = FALSE]
  R  <- if (ncol(Xo) > 0) qr.resid(qr(Xo), Xt) else Xt
  R %*% solve(crossprod(R))
}
