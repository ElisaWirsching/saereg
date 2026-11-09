# Tests of saereg() (2026-11-05): every quantity against an explicit computation with lm(),
# fixest::feols() or sandwich, on small simulated data.

sim <- function(seed = 1) {
  set.seed(seed)
  n <- 120; D <- 8; M <- 4
  W <- apply(matrix(rnorm(D * M), D, M), 2, function(w) w / sqrt(sum(w^2)))   # unit, not orthogonal
  d <- data.frame(id = rep(1:30, each = 4), t = rep(1:4, 30), x = rnorm(n),
                  g = sample(letters[1:5], n, replace = TRUE))
  d$treat <- as.integer(d$id <= 15 & d$t >= 3)                 # treated units after period 2
  d$ever  <- as.integer(d$id <= 15)
  E <- matrix(rnorm(n * D), n, D) + outer(d$treat, W[, 1]) * 0.8 + rnorm(30)[d$id]
  list(d = d, E = E, W = W)
}

test_that("beta is the OLS coefficient and the shares follow from it", {
  s <- sim(); d <- s$d; E <- s$E; W <- s$W
  fit <- saereg(E ~ treat + x + g, data = d, W = W, draws = 19L)
  b_lm <- coef(lm(E ~ treat + x + g, data = d))["treat", ]
  expect_equal(unname(fit$beta[, 1]), unname(b_lm), tolerance = 1e-10)
  a <- drop(crossprod(W, b_lm)); Gi <- solve(crossprod(W))
  expect_equal(fit$features$alpha, unname(a), tolerance = 1e-10)
  expect_equal(fit$features$delta, unname(drop(Gi %*% a)), tolerance = 1e-10)
  expect_equal(sum(fit$features$q_j), fit$joint$R2, tolerance = 1e-12)
  expect_equal(fit$joint$R2, drop(a %*% Gi %*% a) / sum(b_lm^2), tolerance = 1e-10)
})

test_that("without clusters: HC2 bias correction and the closed-form jackknife", {
  s <- sim(2); d <- s$d; E <- s$E; W <- s$W
  fit <- saereg(E ~ treat + x, data = d, W = W, draws = 19L)
  v <- vapply(seq_len(ncol(E)), function(k) {
    sandwich::vcovHC(lm(E[, k] ~ treat + x, data = d), type = "HC2")["treat", "treat"]
  }, numeric(1))
  expect_equal(fit$global$norm_sq_cor, sum(fit$beta^2) - sum(v), tolerance = 1e-10)
  # explicit leave-one-document-out jackknife of alpha and delta
  Gi <- solve(crossprod(W))
  jk <- t(vapply(seq_len(nrow(d)), function(i) {
    a <- drop(crossprod(W, coef(lm(E[-i, ] ~ treat + x, data = d[-i, ]))["treat", ]))
    c(a, drop(Gi %*% a))
  }, numeric(2 * ncol(W))))
  n <- nrow(d); M <- ncol(W)
  se <- sqrt((n - 1) / n * colSums(sweep(jk, 2, colMeans(jk))^2))
  expect_equal(fit$features$alpha_se, se[1:M], tolerance = 1e-8)
  expect_equal(fit$features$delta_se, se[M + 1:M], tolerance = 1e-8)
})

test_that("fixed effects: beta equals feols, the weights sum to zero within each effect", {
  s <- sim(3); d <- s$d; E <- s$E; W <- s$W
  fit <- saereg(E ~ treat + x | id + t, data = d, W = W, cluster = ~ id,
                test = "bootstrap", draws = 9L)
  b_fe <- vapply(seq_len(ncol(E)), function(k) {
    coef(fixest::feols(y ~ treat + x | id + t, data = transform(d, y = E[, k])))[["treat"]]
  }, numeric(1))
  expect_equal(unname(fit$beta[, 1]), b_fe, tolerance = 1e-8)
  w <- drop(fwl_weights(cbind(treat = d$treat, x = d$x), "treat", list(d$id, d$t)))
  expect_lt(max(abs(tapply(w, d$id, sum))), 1e-10)
  expect_lt(max(abs(tapply(w, d$t, sum))), 1e-10)
  expect_equal(sum(w * d$treat), 1, tolerance = 1e-10)
})

test_that("with clusters: the jackknife equals refits leaving out one cluster", {
  s <- sim(4); d <- s$d; E <- s$E; W <- s$W
  fit <- saereg(E ~ treat | id + t, data = d, W = W, cluster = ~ id, test = "bootstrap",
                draws = 9L)
  ids <- sort(unique(d$id)); G <- length(ids)
  J <- t(vapply(ids, function(g) {
    keep <- d$id != g
    vapply(seq_len(ncol(E)), function(k) {
      coef(fixest::feols(y ~ treat | id + t, data = transform(d[keep, ], y = E[keep, k])))[["treat"]]
    }, numeric(1))
  }, numeric(ncol(E))))
  jv <- function(Z) (G - 1) / G * colSums(sweep(Z, 2, colMeans(Z))^2)
  AJ <- J %*% W
  expect_equal(fit$features$alpha_se, sqrt(jv(AJ)), tolerance = 1e-7)
  expect_equal(fit$features$delta_se, sqrt(jv(AJ %*% solve(crossprod(W)))), tolerance = 1e-7)
  expect_equal(fit$global$norm_sq_cor, sum(fit$beta^2) - sum(jv(J)), tolerance = 1e-8)
})

test_that("p = (1 + #null >= observed) / (draws + 1); draw b does not depend on the count", {
  s <- sim(5); d <- s$d; E <- s$E; W <- s$W
  for (tst in c("permutation", "bootstrap")) {
    cl <- if (tst == "bootstrap") ~ id else NULL
    st <- if (tst == "bootstrap") ~ ever else NULL
    f19 <- saereg(E ~ treat, data = d, W = W, cluster = cl, strata = st, test = tst, draws = 19L)
    f39 <- saereg(E ~ treat, data = d, W = W, cluster = cl, strata = st, test = tst, draws = 39L)
    expect_equal(f39$null[1:19, ], f19$null[, 1])
    expect_equal(f19$global$p, (1 + sum(f19$null >= f19$global$norm_sq_raw)) / 20)
  }
})

test_that("several terms are estimated jointly, as lm does", {
  s <- sim(6); d <- s$d; E <- s$E; W <- s$W
  d$t_a <- d$treat * (d$id %% 2); d$t_b <- d$treat * (1 - d$id %% 2)
  fit <- saereg(E ~ t_a + t_b + x, data = d, W = W, term = c("t_a", "t_b"), draws = 9L)
  b_lm <- coef(lm(E ~ t_a + t_b + x, data = d))
  expect_equal(unname(fit$beta), unname(t(b_lm[c("t_a", "t_b"), ])), tolerance = 1e-10)
  expect_equal(nrow(fit$features), 2L * ncol(W))
  expect_named(coef(fit), paste0(rep(c("t_a", "t_b"), each = 4), ":f", 0:3))
})

test_that("rows: shared embeddings give the same fit as the expanded matrix", {
  s <- sim(7); d <- s$d; E <- s$E; W <- s$W
  r <- c(seq_len(nrow(d)), seq_len(nrow(d)))                     # every embedding used twice
  d2 <- rbind(d, transform(d, id = id + 100L))                   # second "stack"
  d2$cl <- d2$id %% 100
  E2 <- E[r, ]
  a <- saereg(E ~ treat | id + t, data = d2, W = W, cluster = ~ cl, rows = r,
              test = "bootstrap", draws = 9L)
  b <- saereg(E2 ~ treat | id + t, data = d2, W = W, cluster = ~ cl, test = "bootstrap",
              draws = 9L)
  expect_equal(a$beta, b$beta, tolerance = 1e-10)
  expect_equal(a$features$delta_se, b$features$delta_se, tolerance = 1e-10)
  expect_equal(a$null, b$null, tolerance = 1e-10)
})

test_that("activations: gamma is the term's coefficient on a_ij", {
  s <- sim(8); d <- s$d; E <- s$E; W <- s$W
  A <- pmax(E %*% W, 0)
  fit <- saereg(E ~ treat + x, data = d, W = W, activations = A, draws = 9L)
  expect_equal(fit$features$gamma, unname(coef(lm(A ~ treat + x, data = d))["treat", ]),
               tolerance = 1e-10)
  expect_equal(fit$features$on_shift,
               unname(coef(lm((A > 0) * 1 ~ treat + x, data = d))["treat", ]), tolerance = 1e-10)
  expect_equal(fit$features$share_on, unname(colMeans(A > 0)))
})

test_that("designs the methods do not cover stop with a message", {
  s <- sim(9); d <- s$d; E <- s$E; W <- s$W
  expect_error(saereg(E ~ treat | id, data = d, W = W), "need `cluster`")
  expect_error(saereg(E ~ treat, data = d, W = W, cluster = ~ id), "bootstrap")
  expect_error(saereg(E ~ treat, data = d, W = W * 2), "unit length")
})
