# Depuis la racine : Rscript tests/test_data_acn.R
library(testthat)
source("src/data_acn.R")
test_that("sessions ACN chargées et cohérentes", {
  d <- load_acn("office001")
  expect_gt(nrow(d), 100)
  expect_true(all(d$disconnectTime >= d$connectionTime, na.rm = TRUE))
  expect_true(any(!is.na(d$kWhRequested)))
  expect_false(anyDuplicated(d$sessionID) > 0)
})
