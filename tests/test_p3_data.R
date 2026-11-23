# Depuis la racine : Rscript tests/test_p3_data.R
library(testthat)
source("src/p3_data.R")
x <- build_p3()
test_that("cible et découpage cohérents", {
  expect_equal(x$early, as.integer(x$gap < -P3_SEUIL))
  expect_true(all(x$split[x$conn < P3_TRAIN_END] == "train"))
  expect_true(max(x$conn[x$split == "val"]) < min(x$conn[x$split == "test"]))
  expect_false(any(c("disconnectTime", "doneChargingTime", "kWhDelivered", "gap", "dur_real") %in% P3_FEATURES))
  expect_false(anyNA(x[, P3_FEATURES]))
})
test_that("historique usager sans fuite (sessions terminées avant l'arrivée)", {
  set.seed(1)
  for (i in sample(nrow(x), 200)) {
    p <- x[x$userID == x$userID[i] & x$disc < x$conn[i], ]
    expect_equal(x$n_past[i], nrow(p))
    expect_equal(x$past_rate[i], (sum(p$early) + P3_PRIOR_M * mean(x$early[x$split == "train"])) / (nrow(p) + P3_PRIOR_M))
  }
})
