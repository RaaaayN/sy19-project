# Depuis la racine : Rscript tests/test_features.R
library(testthat)
source("src/features.R")
test_that("variables par usager cohérentes et split sans fuite", {
  L <- load_users()
  expect_true(all(L$u$n >= MIN_SESSIONS))
  expect_false(anyDuplicated(L$u$userID) > 0)
  expect_length(intersect(L$tr$userID, L$te$userID), 0)
  expect_true(all(L$u$arr_med >= 0 & L$u$arr_med < 24))
  expect_true(all(L$u$reg_weeks > 0 & L$u$reg_weeks <= 1))
  expect_equal(unname(colMeans(L$ztr)), rep(0, length(CLUST_VARS)), tolerance = 1e-8)
  expect_false(isTRUE(all.equal(unname(colMeans(L$zte)), rep(0, length(CLUST_VARS)))))  # échelle du train
  expect_false(anyNA(L$ztr)); expect_false(anyNA(L$zte))
})
