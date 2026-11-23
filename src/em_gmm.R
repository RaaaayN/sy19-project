# Mélange de gaussiennes (covariances pleines) estimé par EM, en R de base (matrices uniquement).
# fit <- em_gmm(X, K); predict_gmm(fit, Xnew); rgmm(n, fit)

# log-densité gaussienne multivariée de chaque ligne de X, via Cholesky
ldmvn <- function(X, mu, S) {
  R <- chol(S)
  Q <- backsolve(R, t(X) - mu, transpose = TRUE)   # R^T q = (x - mu)
  -0.5 * colSums(Q^2) - sum(log(diag(R))) - 0.5 * ncol(X) * log(2 * pi)
}

# log p(x_i, z_i = k) pour tous i, k
.logjoint <- function(X, fit)
  sapply(seq_along(fit$pi), function(k) log(fit$pi[k]) + ldmvn(X, fit$mu[k, ], fit$Sigma[[k]]))

.lse <- function(M) { m <- apply(M, 1, max); m + log(rowSums(exp(M - m))) }

gmm_npar <- function(K, p) (K - 1) + K * p + K * p * (p + 1) / 2

em_gmm <- function(X, K, n_init = 10, max_iter = 500, tol = 1e-7, reg = 1e-6, seed = 19) {
  X <- as.matrix(X); n <- nrow(X); p <- ncol(X)
  set.seed(seed)
  best <- NULL
  for (init in seq_len(n_init)) {
    # Initialisation : partition aléatoire par k-means (une seule graine par essai)
    cl <- if (K == 1) rep(1, n) else kmeans(X, K, nstart = 1, iter.max = 50)$cluster
    Rsp <- outer(cl, seq_len(K), "==") * 1
    ll_old <- -Inf
    for (it in seq_len(max_iter)) {
      # M-step
      nk <- colSums(Rsp)
      if (any(nk < p + 1)) break                       # composante dégénérée : essai abandonné
      fit <- list(pi = nk / n, mu = t(Rsp) %*% X / nk, Sigma = vector("list", K))
      for (k in seq_len(K)) {
        Xc <- sweep(X, 2, fit$mu[k, ])
        fit$Sigma[[k]] <- crossprod(Xc * Rsp[, k], Xc) / nk[k] + diag(reg, p)
      }
      # E-step (log-sum-exp pour la stabilité numérique)
      LJ <- .logjoint(X, fit); lse <- .lse(LJ)
      Rsp <- exp(LJ - lse); ll <- sum(lse)
      if (ll - ll_old < tol * abs(ll)) break
      ll_old <- ll
    }
    if (any(colSums(Rsp) < p + 1)) next
    if (is.null(best) || ll > best$loglik) { fit$loglik <- ll; fit$iter <- it; fit$resp <- Rsp; best <- fit }
  }
  if (is.null(best)) return(NULL)
  k <- gmm_npar(K, p)
  best$cluster <- max.col(best$resp, ties.method = "first")
  best$npar <- k; best$n <- n
  best$bic <- -2 * best$loglik + k * log(n)
  best$aic <- -2 * best$loglik + 2 * k
  best
}

# Responsabilités, classes et log-vraisemblance de nouvelles données
predict_gmm <- function(fit, X) {
  LJ <- .logjoint(as.matrix(X), fit); lse <- .lse(LJ)
  R <- exp(LJ - lse)
  list(resp = R, cluster = max.col(R, ties.method = "first"), loglik = sum(lse), ll_i = lse)
}

# Tirage de n points selon le mélange
rgmm <- function(n, fit) {
  z <- sample(seq_along(fit$pi), n, replace = TRUE, prob = fit$pi)
  p <- ncol(fit$mu); X <- matrix(0, n, p)
  for (k in unique(z)) {
    i <- which(z == k)
    X[i, ] <- matrix(rnorm(length(i) * p), ncol = p) %*% chol(fit$Sigma[[k]]) +
      matrix(fit$mu[k, ], length(i), p, byrow = TRUE)
  }
  colnames(X) <- colnames(fit$mu)
  attr(X, "component") <- z
  X
}
