# Depuis la racine : Rscript tests/test_depart_risk.R (après Rscript src/p3_export.R)
library(testthat)
source("src/p3_data.R"); source("src/depart_risk.R")
test_that("depart_risk reproduit le GAM exporté et renvoie des probabilités", {
  x <- build_p3(); te <- x[x$split == "test", ]
  p <- depart_risk(newdata = te)
  ref <- readRDS("reports/preds/p3_export_gam_test.rds")
  expect_equal(ref$sessionID, te$sessionID)
  expect_lt(max(abs(p - ref$p_gam)), 0.01)            # écart dû à l'interpolation des grilles
  expect_true(all(p > 0 & p < 1))
})
test_that("le risque croît avec le taux historique de l'usager ; nouvel usager géré", {
  p <- depart_risk(site = "jpl", arr_hour = 8, dep_hour = 17, dur_decl = 9, n_past = 20, past_rate = c(0.05, 0.5, 0.9))
  expect_true(all(diff(p) > 0))
  expect_length(depart_risk(site = "caltech", arr_hour = 9, dep_hour = 12, dur_decl = 3), 1)
  expect_equal(marge_llf(0), 0)
})
