# Depuis la racine : Rscript tests/test_features.R
library(testthat)
source("src/features.R")

test_that("historique usager : seules les sessions terminées avant l'arrivée comptent", {
  t0 <- as.POSIXct("2019-01-01 08:00", tz = "UTC")
  conn <- t0 + c(0, 1, 24, 48) * 3600
  disc <- conn + c(30, 2, 5, 5) * 3600        # la 1re session se termine après l'arrivée de la 3e
  h <- user_history(conn, disc, dur = c(30, 2, 5, 5), kwh = c(10, 2, 4, 6), hour = c(8, 9, 8, 8))
  expect_equal(h$h_n, c(0, 0, 1, 3))
  expect_equal(h$h_dur_mean, c(NA, NA, 2, 37 / 3))  # session 3 : seule la 2e est finie
  expect_equal(h$h_last_kwh, c(NA, NA, 2, 10))  # dernière terminée = 1re (fin à +30 h)
})

test_that("sessions nettoyées et découpage temporel sans recouvrement", {
  s <- load_sessions()
  expect_gt(nrow(s), 40000)
  expect_true(all(s$duration > 0 & s$duration <= 48))
  expect_true(max(s$conn[s$split == "train"]) < min(s$conn[s$split == "val"]))
  expect_true(max(s$conn[s$split == "val"]) < min(s$conn[s$split == "test"]))
  expect_true(all(s$has_hist[s$known == 0] == 0))
  expect_false(anyNA(s[, ALL_VARS]))
})
