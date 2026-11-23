# Étape 10 : générateur de véhicules pour le simulateur IA04.
# Pour chaque profil d'usager, un GMM (EM maison) est appris sur les sessions des usagers du train dans
# l'espace (heure d'arrivée, log durée, log énergie), K choisi par BIC. Les paramètres sont exportés en JSON.
# Usage : Rscript src/generate_agents.R                      # apprend et écrit reports/results/agents_gmm.json
#         source("src/generate_agents.R"); P <- read_agents_params(); v <- generate_agents(1000, P)
suppressPackageStartupMessages({ library(jsonlite); library(dplyr) })
source("src/features.R"); source("src/em_gmm.R")
AGENTS_JSON <- "reports/results/agents_gmm.json"
SESSION_VARS <- c("arrival_hour", "log_duration_h", "log_energy_kwh")

session_matrix <- function(s) cbind(arrival_hour = s$arr_h, log_duration_h = log(s$dur_h), log_energy_kwh = log(s$kWhDelivered))

# Sessions des usagers d'un jeu (train/test), avec leur profil
profile_sessions <- function(jeu = c("train", "test")) {
  jeu <- match.arg(jeu); pr <- readRDS("reports/results/profils.rds")
  inner_join(clean_sessions(load_acn()), pr[[if (jeu == "train") "tr" else "te"]], by = "userID")
}

read_agents_params <- function(f = AGENTS_JSON) fromJSON(f, simplifyVector = FALSE)

# Tire n véhicules (sessions) : profil selon la part des sessions, puis composante, puis vecteur gaussien (rejet hors bornes)
generate_agents <- function(n, P = read_agents_params(), seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  w <- sapply(P$profiles, `[[`, "session_share")
  prof <- sample(seq_along(w), n, replace = TRUE, prob = w)
  out <- lapply(seq_along(w), function(j) {
    nj <- sum(prof == j); if (nj == 0) return(NULL)
    comps <- P$profiles[[j]]$components
    fit <- list(pi = sapply(comps, `[[`, "weight"),
                mu = do.call(rbind, lapply(comps, function(c) unlist(c$mean))),
                Sigma = lapply(comps, function(c) do.call(rbind, lapply(c$cov, unlist))))
    colnames(fit$mu) <- SESSION_VARS
    x <- matrix(nrow = 0, ncol = 3)
    while (nrow(x) < nj) {
      y <- rgmm(2 * nj, fit)
      ok <- y[, 1] >= 0 & y[, 1] < 24 & y[, 2] >= log(0.1) & y[, 2] <= log(48)   # mêmes bornes que les données
      x <- rbind(x, y[ok, , drop = FALSE])
    }
    x <- x[seq_len(nj), , drop = FALSE]
    data.frame(profile = P$profiles[[j]]$name, arrival_h = x[, 1],
               departure_h = x[, 1] + exp(x[, 2]), energy_kwh = exp(x[, 3]))
  })
  do.call(rbind, out)[order(sample(n)), ]
}

if (sys.nframe() == 0) {
  s <- profile_sessions("train"); pr <- readRDS("reports/results/profils.rds")
  pr$tr <- merge(pr$tr, load_users()$tr[, c("userID", "freq_week")], by = "userID")
  sel <- list(); KS <- 1:8
  profiles <- lapply(seq_along(pr$noms), function(j) {
    X <- session_matrix(s[s$profil == pr$noms[j], ])
    fits <- lapply(KS, function(K) em_gmm(X, K, n_init = 3, seed = SEED))
    bic <- sapply(fits, function(f) if (is.null(f)) NA else f$bic)
    # Avec ~10 000 sessions le BIC décroît encore à K = 8 : règle du coude, premier K tel que passer
    # à K + 1 améliore le BIC de moins de 1 % (JSON lisible, pas de surajustement)
    gain <- -diff(bic) / abs(bic[-length(bic)])
    K <- if (any(gain < 0.01, na.rm = TRUE)) which(gain < 0.01)[1] else max(KS)
    f <- fits[[K]]
    sel[[j]] <<- data.frame(profil = pr$noms[j], sessions = nrow(X), K = KS, BIC = round(bic, 1), retenu = KS == K)
    cat(pr$noms[j], ": sessions", nrow(X), "| K min BIC =", which.min(bic), "| K retenu =", K, "\n")
    u <- pr$tr[pr$tr$profil == pr$noms[j], ]
    list(name = pr$noms[j],
         weight = round(mean(pr$tr$profil == pr$noms[j]), 4),       # part des usagers (types d'agents)
         session_share = round(nrow(X) / nrow(s), 4),               # part des sessions (tirage des véhicules)
         sessions_per_week_median = round(median(u$freq_week), 3),
         components = lapply(seq_along(f$pi), function(k) list(
           weight = f$pi[k], mean = unname(f$mu[k, ]), cov = unname(lapply(1:3, function(i) f$Sigma[[k]][i, ])))))
  })
  P <- list(
    description = "Profils d'usagers ACN-Data (SY19 projet 2). Par profil, mélange gaussien des sessions.",
    timezone = "America/Los_Angeles",
    variables = SESSION_VARS,
    transform = "departure_h = arrival_h + exp(log_duration_h) ; energy_kwh = exp(log_energy_kwh) ; rejeter arrival_h hors [0,24) et durée hors [0,1 h ; 48 h]",
    profiles = profiles)
  write_json(P, AGENTS_JSON, auto_unbox = TRUE, digits = 8, pretty = TRUE)
  write.csv(do.call(rbind, sel), "reports/results/agents_choix_K.csv", row.names = FALSE)
  cat("Paramètres exportés dans", AGENTS_JSON, "\n")
}
