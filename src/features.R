# Variables par usager (projet 2 : profils d'usagers).
# Usage : source("src/features.R"); s <- clean_sessions(load_acn()); u <- user_features(s, 10)
#         Rscript src/features.R   # écrit reports/results/users_features.csv
suppressPackageStartupMessages({ library(dplyr); library(lubridate) })
if (!exists("load_acn")) source("src/data_acn.R")

MIN_SESSIONS <- 10
SEED <- 19
# Variables utilisées pour le clustering (après transformation, voir clust_matrix)
CLUST_VARS <- c("arr_med", "arr_sd", "dep_med", "dur_med", "kwh_med", "kwh_cv",
                "freq_week", "reg_weeks", "frac_weekend")

# Sessions identifiées, heures locales, durées plausibles
clean_sessions <- function(d) {
  d %>%
    filter(!is.na(userID), !is.na(disconnectTime), kWhDelivered > 0) %>%
    mutate(conn = with_tz(connectionTime, "America/Los_Angeles"),
           disc = with_tz(disconnectTime, "America/Los_Angeles"),
           arr_h = hour(conn) + minute(conn) / 60,
           dur_h = as.numeric(difftime(disc, conn, units = "hours")),
           dep_h = arr_h + dur_h,                      # peut dépasser 24 (départ le lendemain)
           weekend = wday(conn, week_start = 1) >= 6,
           day = as.Date(conn, tz = "America/Los_Angeles")) %>%
    filter(dur_h >= 0.1, dur_h <= 48)
}

user_features <- function(s, min_sessions = MIN_SESSIONS) {
  s %>%
    group_by(userID) %>%
    filter(n() >= min_sessions) %>%
    summarise(
      n = n(),
      site = names(which.max(table(site))),
      arr_med = median(arr_h), arr_sd = sd(arr_h),
      dep_med = median(dep_h), dur_med = median(dur_h),
      kwh_med = median(kWhDelivered), kwh_cv = sd(kWhDelivered) / mean(kWhDelivered),
      span_w = max(1, as.numeric(max(day) - min(day)) / 7),
      freq_week = n / span_w,
      # Proportions lissées (+0,5 / +1) : évite les valeurs exactement 0 ou 1 (55 % des usagers
      # n'ont aucune session le week-end), qui rendent les covariances des GMM singulières
      reg_weeks = (n_distinct(floor_date(day, "week")) + 0.5) / (ceiling(span_w) + 1),
      frac_weekend = (sum(weekend) + 0.5) / (n + 1),
      first = min(day), last = max(day),
      .groups = "drop") %>%
    mutate(reg_weeks = pmin(reg_weeks, 1 - 0.5 / (ceiling(span_w) + 1)))
}

# Séparation train/test par usagers (20 % des usagers mis de côté, jamais vus à l'apprentissage)
split_users <- function(u, frac_test = 0.2, seed = SEED) {
  set.seed(seed)
  u$split <- ifelse(seq_len(nrow(u)) %in% sample(nrow(u), round(frac_test * nrow(u))), "test", "train")
  u
}

# Matrice de clustering : log des variables asymétriques, standardisation apprise sur le train
clust_matrix <- function(u, center = NULL, scale = NULL) {
  x <- as.matrix(u[, CLUST_VARS])
  x[, c("kwh_med", "freq_week", "dur_med")] <- log(x[, c("kwh_med", "freq_week", "dur_med")])
  x[, "kwh_cv"] <- log1p(x[, "kwh_cv"])
  x[, c("reg_weeks", "frac_weekend")] <- qlogis(x[, c("reg_weeks", "frac_weekend")])
  if (is.null(center)) { center <- colMeans(x); scale <- apply(x, 2, sd) }
  z <- sweep(sweep(x, 2, center), 2, scale, "/")
  attr(z, "center") <- center; attr(z, "scl") <- scale
  z
}

# Charge le tableau d'usagers produit par ce script, avec la matrice standardisée train/test
load_users <- function(f = "reports/results/users_features.csv") {
  u <- read.csv(f, stringsAsFactors = FALSE)
  tr <- u[u$split == "train", ]; te <- u[u$split == "test", ]
  ztr <- clust_matrix(tr)
  zte <- clust_matrix(te, attr(ztr, "center"), attr(ztr, "scl"))
  list(u = u, tr = tr, te = te, ztr = ztr, zte = zte)
}

if (sys.nframe() == 0) {
  dir.create("reports/results", recursive = TRUE, showWarnings = FALSE)
  s <- clean_sessions(load_acn())
  n_by_user <- table(s$userID)
  cat("Sessions identifiées et nettoyées :", nrow(s), "| usagers :", length(n_by_user), "\n")
  # Justification du seuil : usagers et sessions conservés, précision relative de l'écart-type estimé
  seuils <- data.frame(seuil = c(1, 3, 5, 10, 20, 30))
  seuils$usagers <- sapply(seuils$seuil, function(m) sum(n_by_user >= m))
  seuils$part_sessions <- round(sapply(seuils$seuil, function(m) sum(n_by_user[n_by_user >= m])) / nrow(s), 3)
  seuils$erreur_rel_sd <- round(ifelse(seuils$seuil > 1, 1 / sqrt(2 * (seuils$seuil - 1)), NA), 2)
  print(seuils); write.csv(seuils, "reports/results/seuils_sessions.csv", row.names = FALSE)
  u <- split_users(user_features(s))
  cat("Usagers retenus (n >=", MIN_SESSIONS, ") :", nrow(u), "| train :", sum(u$split == "train"), "| test :", sum(u$split == "test"), "\n")
  print(summary(u[, CLUST_VARS]))
  write.csv(u, "reports/results/users_features.csv", row.names = FALSE)
}
