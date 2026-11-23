# Projet 3, étape 11a : export du modèle de risque pour le simulateur Go (IA04). Rscript src/p3_export.R
# Le GAM logistique est additif : logit(p) = intercept + somme des termes. Chaque terme lisse est
# exporté comme table (grille, valeur) à interpoler linéairement ; Go n'a besoin que d'additions.
# Modèle final réajusté sur train + validation (mêmes hyperparamètres : REML), évalué sur le test.
suppressMessages({ library(mgcv); library(jsonlite); library(dplyr) })
source("src/p3_eval.R")
x <- build_p3(); x$lnp <- log1p(x$n_past)
fit <- gam(early ~ site + weekend + s(arr_hour, bs = "cc", k = 20) + s(dep_hour, bs = "cc", k = 20) +
             s(dur_decl, k = 10) + s(past_rate, k = 10) + s(lnp, k = 6) + s(past_gap, k = 6) +
             kWhRequested + WhPerMile,
           family = binomial, data = x[x$split != "test", ], method = "REML", select = TRUE,
           knots = list(arr_hour = c(0, 24), dep_hour = c(0, 24)))
te <- x[x$split == "test", ]
p_gam <- as.numeric(predict(fit, te, type = "response"))
cat("GAM (train+val) AUC test =", p3_auc(te$early, p_gam), "\n")

# Grilles de chaque terme lisse : predict(type = "terms") en faisant varier une seule variable.
grids <- list(arr_hour = seq(0, 24, by = 0.25), dep_hour = seq(0, 24, by = 0.25),
              dur_decl = seq(0, 24, by = 0.25), past_rate = seq(0, 1, by = 0.01),
              lnp = seq(0, log1p(max(x$n_past)), length.out = 101), past_gap = seq(-12, 12, by = 0.25))
ref <- x[1, ]
terms <- lapply(names(grids), function(v) {
  nd <- ref[rep(1, length(grids[[v]])), ]; nd[[v]] <- grids[[v]]
  tt <- predict(fit, nd, type = "terms")
  list(variable = v, grille = grids[[v]], valeur = round(as.numeric(tt[, sprintf("s(%s)", v)]), 6))
})
cf <- coef(fit)
p0 <- mean(x$early[x$split != "test"])
early <- x$gap[x$early == 1 & x$split != "test"]
model <- list(
  description = paste("Risque de départ anticipé (> 15 min avant l'heure déclarée), GAM logistique.",
                      "logit = intercept + lineaire + somme des termes interpolés ; p = 1/(1+exp(-logit)).",
                      "Bornes : en dehors de la grille, prendre la valeur extrême."),
  seuil_minutes = P3_SEUIL,
  intercept = unname(cf["(Intercept)"]),
  lineaire = list(site_jpl = unname(cf["sitejpl"]), weekend = unname(cf["weekend"]),
                  kWhRequested = unname(cf["kWhRequested"]), WhPerMile = unname(cf["WhPerMile"])),
  termes = terms,
  entrees = c(arr_hour = "heure locale d'arrivée (h décimales)", dep_hour = "heure locale de départ déclarée",
              dur_decl = "durée déclarée (h)", past_rate = "taux lissé de départs anticipés passés de l'usager",
              lnp = "log(1 + nombre de sessions passées terminées)", past_gap = "écart moyen passé départ réel - déclaré (h)"),
  prior = list(p0 = p0, m = P3_PRIOR_M, formule = "past_rate = (k + m*p0) / (n + m) ; nouvel usager : p0"),
  defauts = list(kWhRequested = median(x$kWhRequested), WhPerMile = median(x$WhPerMile)),
  avance_mediane_si_anticipe_min = -median(early),
  avance_q75_si_anticipe_min = -unname(quantile(early, 0.25)),
  auc_test = p3_auc(te$early, p_gam))
dir.create("reports/export", showWarnings = FALSE)
write_json(model, "reports/export/depart_risk_gam.json", auto_unbox = TRUE, pretty = TRUE, digits = NA)

# État de l'historique de chaque usager à la fin des données (pour initialiser le simulateur).
last <- x %>% group_by(userID) %>% summarise(n = n(), k = sum(early), gap_moy_h = mean(gap) / 60,
                                              site = first(site), .groups = "drop") %>%
  mutate(past_rate = (k + P3_PRIOR_M * p0) / (n + P3_PRIOR_M), lnp = log1p(n))
write.csv(last, "reports/export/depart_risk_usagers.csv", row.names = FALSE)

# Table de probabilités par site x créneau d'arrivée (3 h) x tranche d'ancienneté, sur le test.
tab <- te %>% mutate(p = p_gam, creneau = sprintf("%02dh-%02dh", 3 * (floor(arr_hour) %/% 3), 3 * (floor(arr_hour) %/% 3) + 3),
                     anciennete = cut(n_past, c(-1, 0, 4, 19, Inf), labels = c("0", "1-4", "5-19", "20+"))) %>%
  group_by(site, creneau, anciennete) %>% summarise(sessions = n(), p_moyenne = mean(p), taux_observe = mean(early), .groups = "drop")
write.csv(tab, "reports/export/depart_risk_creneaux.csv", row.names = FALSE)
saveRDS(data.frame(sessionID = te$sessionID, p_gam = p_gam), "reports/preds/p3_export_gam_test.rds")
