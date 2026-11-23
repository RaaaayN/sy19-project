# Prédiction à l'arrivée d'un véhicule avec le MLP exporté (reports/model_mlp.json, lisible depuis Go).
# L'inférence est faite en R pur à partir du JSON : c'est exactement le calcul à reproduire côté simulateur IA04.
#   source("src/predict.R")
#   predict_session("jpl", as.POSIXct("2020-01-15 08:30", tz = TZ_LOCAL), user_id = "000000123",
#                   kwh_requested = 20, minutes_available = 540, past = load_sessions())
source("src/features.R")
suppressMessages(library(jsonlite))
MODEL_JSON <- "reports/model_mlp.json"

load_model <- function(path = MODEL_JSON) fromJSON(path, simplifyMatrix = TRUE)

# Propagation avant : x (n x p) -> sorties dénormalisées (n x 2 : durée en h, énergie en kWh)
mlp_forward <- function(m, x) {
  h <- scale(x, m$x_mean, m$x_sd)
  for (l in seq_along(m$layers$W)) {
    h <- sweep(h %*% t(m$layers$W[[l]]), 2, m$layers$b[[l]], "+")
    if (m$layers$activation[l] == "relu") h <- pmax(h, 0)
  }
  out <- sweep(sweep(h, 2, m$y_sd, "*"), 2, m$y_mean, "+")
  colnames(out) <- m$targets
  out
}

# Construit les variables d'une session à partir des sessions passées de l'usager (past = sortie de
# load_sessions() ou toute table avec userID, conn, disc, duration, energy, hour).
session_features <- function(site, arrival, user_id = NA, kwh_requested = NA, minutes_available = NA, past = NULL) {
  arrival <- with_tz(arrival, TZ_LOCAL)
  hr <- hour(arrival) + minute(arrival) / 60
  x <- data.frame(site = factor(site, levels = ACN_SITES), hour = hr, dow = wday(arrival, week_start = 1),
                  known = as.integer(!is.na(user_id)), duration = 0)
  x$weekend <- as.integer(x$dow >= 6)
  decl_dur <- minutes_available / 60
  x$has_decl <- as.integer(!is.na(kwh_requested) && !is.na(decl_dur) && kwh_requested > 0 && kwh_requested <= 100 &&
                           decl_dur > 0 && decl_dur <= 48)
  x$decl_kwh <- if (x$has_decl == 1) kwh_requested else 0
  x$decl_dur <- if (x$has_decl == 1) decl_dur else 0
  p <- if (is.na(user_id) || is.null(past)) past[0, ] else past[which(past$userID == user_id), ]
  h <- if (is.null(p) || nrow(p) == 0) as.data.frame(as.list(setNames(c(0, rep(NA, length(HIST_VARS) - 1)), HIST_VARS))) else {
    # la session courante est ajoutée avec une fin infinie : son historique ne contient que le passé terminé
    hh <- user_history(c(p$conn, arrival), c(p$disc, arrival + 1e9), c(p$duration, 0), c(p$energy, 0), c(p$hour, hr))
    hh[nrow(hh), ]
  }
  x <- cbind(x, h)
  x$has_hist <- as.integer(x$h_n > 0)
  for (v in HIST_VARS) x[[v]][is.na(x[[v]])] <- 0
  x$h_n <- log1p(x$h_n)
  x
}

predict_session <- function(site, arrival, user_id = NA, kwh_requested = NA, minutes_available = NA,
                            past = NULL, model = load_model()) {
  x <- session_features(site, arrival, user_id, kwh_requested, minutes_available, past)
  mm <- model.matrix(model_formula("duration"), x)[, model$features, drop = FALSE]
  as.data.frame(mlp_forward(model, mm))
}

if (sys.nframe() == 0) {
  s <- load_sessions(); u <- s$userID[which(!is.na(s$userID))[1]]
  print(predict_session("jpl", as.POSIXct("2020-01-15 08:30", tz = TZ_LOCAL), user_id = u,
                        kwh_requested = 20, minutes_available = 540, past = s))
  print(predict_session("caltech", as.POSIXct("2020-01-18 14:00", tz = TZ_LOCAL)))
}
