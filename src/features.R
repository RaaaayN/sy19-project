# Variables du projet 1 (durée et énergie prédites à l'arrivée).
# Usage : source("src/features.R"); s <- load_sessions()   # data.frame avec colonne split
source("src/data_acn.R")
suppressMessages({ library(dplyr); library(lubridate) })

TZ_LOCAL    <- "America/Los_Angeles"
SPLIT_VAL   <- as.POSIXct("2019-09-01", tz = TZ_LOCAL)  # train : sept. 2018 -> août 2019
SPLIT_TEST  <- as.POSIXct("2019-12-01", tz = TZ_LOCAL)  # val : sept. -> nov. 2019, test : déc. 2019 -> mars 2020
TARGETS     <- c("duration", "energy")
HIST_VARS   <- c("h_n", "h_dur_mean", "h_dur_sd", "h_kwh_mean", "h_kwh_sd", "h_hour_mean", "h_last_dur", "h_last_kwh")
DECL_VARS   <- c("has_decl", "decl_kwh", "decl_dur")
BASE_VARS   <- c("site", "hour", "dow", "weekend", "known")
ALL_VARS    <- c(BASE_VARS, "has_hist", HIST_VARS, DECL_VARS)

# Historique strictement passé : pour une session arrivant en t, on ne garde que les
# sessions du même usager déjà terminées (disconnectTime < t), donc observables en t.
user_history <- function(conn, disc, dur, kwh, hour) {
  o  <- order(disc); dd <- disc[o]
  k  <- findInterval(as.numeric(conn), as.numeric(dd), left.open = TRUE)  # nb de sessions finies avant t
  cs <- function(x) c(0, cumsum(x[o]))
  m  <- function(x) ifelse(k > 0, cs(x)[k + 1] / k, NA)
  sdv <- function(x) {
    v <- (cs(x^2)[k + 1] - k * m(x)^2) / pmax(k - 1, 1)
    ifelse(k > 1, sqrt(pmax(v, 0)), NA)
  }
  last <- function(x) ifelse(k > 0, x[o][pmax(k, 1)], NA)
  data.frame(h_n = k, h_dur_mean = m(dur), h_dur_sd = sdv(dur), h_kwh_mean = m(kwh),
             h_kwh_sd = sdv(kwh), h_hour_mean = m(hour), h_last_dur = last(dur), h_last_kwh = last(kwh))
}

load_sessions <- function() {
  d <- load_acn()
  conn <- with_tz(d$connectionTime, TZ_LOCAL)
  s <- data.frame(
    sessionID = d$sessionID, site = factor(d$site), userID = d$userID, conn = conn,
    disc      = with_tz(d$disconnectTime, TZ_LOCAL),
    duration  = as.numeric(difftime(d$disconnectTime, d$connectionTime, units = "hours")),
    energy    = d$kWhDelivered,
    hour      = hour(conn) + minute(conn) / 60,
    dow       = wday(conn, week_start = 1),            # 1 = lundi
    known     = as.integer(!is.na(d$userID)),
    decl_kwh  = d$kWhRequested,
    decl_dur  = d$minutesAvailable / 60
  )
  s$weekend <- as.integer(s$dow >= 6)
  # Nettoyage : sessions physiquement plausibles (durée entre 5 min et 48 h, énergie > 0.5 kWh
  # déjà garanti par l'API) ; déclarations aberrantes mises à NA.
  s <- s[!is.na(s$duration) & s$duration > 1/12 & s$duration <= 48 & s$energy > 0, ]
  s$decl_kwh[!is.na(s$decl_kwh) & (s$decl_kwh <= 0 | s$decl_kwh > 100)] <- NA
  s$decl_dur[!is.na(s$decl_dur) & (s$decl_dur <= 0 | s$decl_dur > 48)] <- NA
  s$has_decl <- as.integer(!is.na(s$decl_kwh) & !is.na(s$decl_dur))
  s$decl_kwh[s$has_decl == 0] <- 0; s$decl_dur[s$has_decl == 0] <- 0

  # Historique par usager (calculé sur toute la période : seul le passé de chaque session est utilisé)
  h <- as.data.frame(matrix(NA_real_, nrow(s), length(HIST_VARS), dimnames = list(NULL, HIST_VARS)))
  h$h_n <- 0
  for (idx in split(seq_len(nrow(s)), s$userID)) {   # split ignore les userID NA
    h[idx, ] <- with(s[idx, ], user_history(conn, disc, duration, energy, hour))
  }
  s <- cbind(s, h)
  s$has_hist <- as.integer(s$h_n > 0)
  # Sans historique : 0 + indicateur has_hist (un écart-type absent avec 1 seule session vaut 0)
  for (v in HIST_VARS) s[[v]][is.na(s[[v]])] <- 0
  s$h_n <- log1p(s$h_n)   # nombre de sessions passées, en échelle log

  s$split <- factor(ifelse(s$conn < SPLIT_VAL, "train", ifelse(s$conn < SPLIT_TEST, "val", "test")),
                    levels = c("train", "val", "test"))
  s[order(s$conn), ]
}

# Métriques communes
metrics <- function(y, p) c(MAE = mean(abs(y - p)), RMSE = sqrt(mean((y - p)^2)),
                            R2 = 1 - sum((y - p)^2) / sum((y - mean(y))^2))

# Enregistre les métriques val/test d'un modèle dans reports/results/<fichier>.csv
save_results <- function(rows, file) {
  dir.create("reports/results", showWarnings = FALSE, recursive = TRUE)
  write.csv(rows, file.path("reports/results", file), row.names = FALSE)
  print(rows, digits = 4)
}

# Une ligne de résultats par (modèle, cible, split) à partir de prédictions
eval_rows <- function(model, target, s, pred, splits = c("val", "test")) {
  do.call(rbind, lapply(splits, function(sp) {
    i <- s$split == sp
    data.frame(model = model, target = target, split = sp, t(metrics(s[[target]][i], pred[i])))
  }))
}

# Formule commune (toutes les variables)
model_formula <- function(target, vars = ALL_VARS) reformulate(vars, response = target)

if (sys.nframe() == 0) {
  s <- load_sessions()
  print(table(s$split)); print(summary(s[, c(TARGETS, HIST_VARS, DECL_VARS)]))
}
