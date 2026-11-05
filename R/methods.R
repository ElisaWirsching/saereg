#' @export
print.saereg <- function(x, ...) {
  cat("SAE regression:", deparse(x$formula), "\n")
  cat(sprintf("N = %d%s | %d directions | global test: %s, %d draws\n", x$n,
              if (is.na(x$n_clusters)) "" else sprintf(" (%d clusters)", x$n_clusters),
              length(unique(x$features$feature)), x$test, x$draws))
  for (k in seq_len(nrow(x$global))) {
    g <- x$global[k, ]; j <- x$joint[k, ]
    cat(sprintf(paste0("%s: ||beta||^2 %.4g (corrected %.4g, %.0f%% CI %.4g to %.4g), p = %.4g",
                       " | R^2 %.3f (corrected %.3f)\n"),
                g$term, g$norm_sq_raw, g$norm_sq_cor, 100 * x$level, g$ci_lo, g$ci_hi, g$p,
                j$R2, j$R2_cor))
  }
  invisible(x)
}

#' Summary of an SAE regression
#'
#' The global test and, per term, the directions ordered by their bias-corrected share
#' `q_j_cor`, with delta and its interval and the label. The printed table leaves out alpha
#' to give the labels room; `summary(fit)$table` and `fit$features` hold all columns.
#'
#' @param object An `saereg` fit.
#' @param n Directions shown per term (all by default).
#' @param ... Unused.
#' @export
summary.saereg <- function(object, n = Inf, ...) {
  ft <- object$features
  ft <- ft[order(ft$term, -ft$q_j_cor), , drop = FALSE]
  ft <- do.call(rbind, lapply(split(ft, ft$term), utils::head, n = n))
  structure(list(fit = object, table = ft), class = "summary.saereg")
}

#' @export
print.summary.saereg <- function(x, ...) {
  print(x$fit)
  tb <- x$table
  show <- data.frame(term = tb$term, feature = tb$feature,
                     q_j_cor = sprintf("%.3f", tb$q_j_cor),
                     delta = sprintf("%.3f [%.3f, %.3f]", tb$delta, tb$delta_lo, tb$delta_hi))
  if (length(unique(show$term)) == 1L) show$term <- NULL
  # the label last, cut to the width that is left
  used <- sum(mapply(function(v, nm) max(nchar(c(nm, v))), show, names(show)))
  room <- max(20L, getOption("width") - used - ncol(show) - 7L)
  lab <- ifelse(is.na(tb$label), "", tb$label)
  show$label <- ifelse(nchar(lab) > room, paste0(substr(lab, 1, room - 3), "..."), lab)
  cat("\n")
  print(show, row.names = FALSE, right = FALSE)
  invisible(x)
}

#' Coefficients and intervals of an SAE regression
#'
#' `coef()` returns delta (the effect along each direction, the others held fixed; default)
#' or alpha (the simple projection); `confint()` their jackknife intervals at any level.
#'
#' @param object An `saereg` fit.
#' @param type `"delta"` or `"alpha"`.
#' @param parm Directions (feature numbers, from 0) to return; all by default.
#' @param level Confidence level.
#' @param ... Unused.
#' @export
coef.saereg <- function(object, type = c("delta", "alpha"), ...) {
  type <- match.arg(type)
  ft <- object$features
  stats::setNames(ft[[type]], coef_names(ft))
}

#' @rdname coef.saereg
#' @export
confint.saereg <- function(object, parm = NULL, level = object$level,
                           type = c("delta", "alpha"), ...) {
  type <- match.arg(type)
  ft <- object$features
  if (!is.null(parm)) ft <- ft[ft$feature %in% parm, , drop = FALSE]
  z <- stats::qnorm(1 - (1 - level) / 2)
  se <- ft[[paste0(type, "_se")]]
  out <- cbind(ft[[type]] - z * se, ft[[type]] + z * se)
  dimnames(out) <- list(coef_names(ft),
                        paste(format(100 * c((1 - level) / 2, 1 - (1 - level) / 2)), "%"))
  out
}

# Names "f0", "f1", ...; prefixed by the term when a fit has several.
coef_names <- function(ft) {
  nm <- paste0("f", ft$feature)
  if (length(unique(ft$term)) > 1L) paste0(ft$term, ":", nm) else nm
}
