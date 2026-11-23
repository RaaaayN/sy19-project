# Depuis la racine : Rscript tests/test_predict.R (après src/export.R)
library(testthat)
source("src/predict.R")

test_that("predict_session reproduit les prédictions exportées (mêmes variables, sans fuite)", {
  s <- load_sessions(); pr <- read.csv("reports/predictions.csv")
  set.seed(1)
  for (i in sample(which(s$split == "test" & s$has_hist == 1 & s$has_decl == 1), 5)) {
    p <- predict_session(as.character(s$site[i]), s$conn[i], s$userID[i], s$decl_kwh[i], s$decl_dur[i] * 60, past = s)
    ref <- pr[pr$sessionID == s$sessionID[i], ]
    expect_equal(p$duration, ref$duration_pred_h, tolerance = 1e-3)
    expect_equal(p$energy, ref$energy_pred_kwh, tolerance = 1e-3)
  }
  p0 <- predict_session("caltech", as.POSIXct("2020-01-18 14:00", tz = TZ_LOCAL))
  expect_true(all(is.finite(unlist(p0))))
})
