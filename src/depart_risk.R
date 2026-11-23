# Projet 3 : probabilité qu'un usager parte plus de 15 min avant l'heure qu'il a déclarée.
# Utilise le modèle exporté (reports/export/depart_risk_gam.json, produit par src/p3_export.R),
# exactement le même calcul que celui à reproduire en Go dans le simulateur IA04.
# Usage :
#   source("src/depart_risk.R")
#   depart_risk(site = "jpl", arr_hour = 8.5, dep_hour = 17, dur_decl = 8.5, weekend = 0,
#               n_past = 12, past_rate = 0.3, past_gap = 0.5)
#   depart_risk(newdata = data.frame(...))   # mêmes colonnes, vectorisé
#   marge_llf(p) : marge de sécurité (min) à retrancher de l'heure déclarée dans le LLF.
suppressMessages(library(jsonlite))

.dr_model <- local({ m <- NULL; function(path = "reports/export/depart_risk_gam.json") {
  if (is.null(m)) m <<- fromJSON(path, simplifyVector = TRUE); m } })

depart_risk <- function(site = "caltech", arr_hour, dep_hour, dur_decl, weekend = 0, n_past = 0,
                        past_rate = NULL, past_gap = 0, kWhRequested = NULL, WhPerMile = NULL,
                        newdata = NULL, model = .dr_model()) {
  if (!is.null(newdata)) return(do.call(depart_risk, c(as.list(newdata[intersect(names(newdata), names(formals(depart_risk)))]),
                                                       list(model = model))))
  if (is.null(past_rate))    past_rate <- model$prior$p0           # nouvel usager : taux moyen
  if (is.null(kWhRequested)) kWhRequested <- model$defauts$kWhRequested
  if (is.null(WhPerMile))    WhPerMile <- model$defauts$WhPerMile
  stopifnot(all(as.character(site) %in% c("caltech", "jpl")))
  v <- list(arr_hour = arr_hour, dep_hour = dep_hour, dur_decl = dur_decl, past_rate = past_rate,
            lnp = log1p(n_past), past_gap = past_gap)
  li <- model$lineaire
  eta <- model$intercept + li$site_jpl * (as.character(site) == "jpl") + li$weekend * weekend +
    li$kWhRequested * kWhRequested + li$WhPerMile * WhPerMile
  tm <- model$termes
  for (j in seq_len(nrow(tm))) {
    eta <- eta + approx(tm$grille[[j]], tm$valeur[[j]], xout = v[[tm$variable[j]]], rule = 2)$y
  }
  as.numeric(1 / (1 + exp(-eta)))
}

# Marge de sécurité pour le LLF : avance attendue = p * avance médiane observée quand l'usager part tôt.
marge_llf <- function(p, model = .dr_model()) p * model$avance_mediane_si_anticipe_min
