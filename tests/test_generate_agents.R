# Depuis la racine : Rscript tests/test_generate_agents.R  (après Rscript src/generate_agents.R)
# Compare les sessions générées aux vraies sessions des usagers du TEST (jamais vus à l'apprentissage).
library(testthat)
source("src/generate_agents.R")
P <- read_agents_params()

test_that("JSON des GMM lisible et valide", {
  expect_equal(unlist(P$variables), SESSION_VARS)
  expect_equal(sum(sapply(P$profiles, `[[`, "weight")), 1, tolerance = 1e-3)
  expect_equal(sum(sapply(P$profiles, `[[`, "session_share")), 1, tolerance = 1e-3)
  for (p in P$profiles) {
    expect_equal(sum(sapply(p$components, `[[`, "weight")), 1, tolerance = 1e-6)
    for (c in p$components) {
      S <- do.call(rbind, lapply(c$cov, unlist))
      expect_equal(S, t(S)); expect_true(all(eigen(S)$values > 0))
      expect_length(unlist(c$mean), 3)
    }
  }
})

ks_table <- function(gen, real) {
  D <- function(a, b) unname(round(suppressWarnings(ks.test(a, b))$statistic, 3))
  one <- function(g, r, nm) data.frame(profil = nm, arrivee = D(g$arrival_h, r$arr_h),
                                       depart = D(g$departure_h, r$dep_h), energie = D(g$energy_kwh, r$kWhDelivered))
  rbind(one(gen, real, "tous"), do.call(rbind, lapply(sapply(P$profiles, `[[`, "name"), function(nm)
    one(gen[gen$profile == nm, ], real[real$profil == nm, ], nm))))
}

test_that("distributions générées proches des vraies (statistique D de Kolmogorov-Smirnov)", {
  gen <- generate_agents(20000, P, seed = 19)
  expect_true(all(gen$departure_h > gen$arrival_h) && all(gen$energy_kwh > 0))
  tr <- profile_sessions("train"); te <- profile_sessions("test")
  # Référence : écart entre sessions réelles du train et du test (variabilité entre usagers)
  tr_as_gen <- data.frame(profile = tr$profil, arrival_h = tr$arr_h, departure_h = tr$dep_h, energy_kwh = tr$kWhDelivered)
  res <- rbind(cbind(jeu = "train", ks_table(gen, tr)), cbind(jeu = "test", ks_table(gen, te)),
               cbind(jeu = "réel train vs réel test", ks_table(tr_as_gen, te)))
  print(res); write.csv(res, "reports/results/ks_generateur.csv", row.names = FALSE)
  # Avec des milliers de sessions, la p-value rejette au moindre écart : on borne la statistique D.
  # Train : le générateur reproduit les sessions qui l'ont appris (vérifie l'export et le tirage).
  expect_true(all(res[res$jeu == "train", 3:5] < 0.07))
  # Test (usagers jamais vus) : écart plus grand, dû à la variabilité entre usagers (93 usagers)
  expect_true(all(res[res$jeu == "test" & res$profil == "tous", 3:5] < 0.15))
})
