# Depuis la racine : Rscript tests/test_em_gmm.R
library(testthat); suppressPackageStartupMessages(library(mclust))
source("src/em_gmm.R")
test_that("EM retrouve un mélange simulé et égale mclust", {
  set.seed(1)
  X <- rbind(MASS::mvrnorm(300, c(0, 0), matrix(c(1, .5, .5, 1), 2)),
             MASS::mvrnorm(200, c(5, 3), diag(c(.5, 2))))
  f <- em_gmm(X, 2)
  expect_equal(sort(f$pi), c(.4, .6), tolerance = .03)
  expect_equal(f$mu[order(f$mu[, 1]), ], rbind(c(0, 0), c(5, 3)), tolerance = .2, ignore_attr = TRUE)
  expect_equal(rowSums(f$resp), rep(1, 500))
  # log-densité comparée à la formule directe
  S <- f$Sigma[[1]]; x <- X[1, ]; m <- f$mu[1, ]
  direct <- -0.5 * t(x - m) %*% solve(S) %*% (x - m) - 0.5 * log(det(2 * pi * S))
  expect_equal(ldmvn(X[1, , drop = FALSE], m, S), c(direct))
  m2 <- Mclust(X, G = 2, modelNames = "VVV", verbose = FALSE)
  expect_equal(f$loglik, m2$loglik, tolerance = 1e-3)
  expect_equal(mclust::adjustedRandIndex(f$cluster, m2$classification), 1)
  g <- rgmm(1000, f); expect_equal(dim(g), c(1000, 2))
})
