# Projet 3 : jeu de données "départ anticipé" (fiabilité de l'heure déclarée).
# Usage : source("src/p3_data.R"); x <- build_p3()     # depuis la racine du worktree
#         Rscript src/p3_data.R                         # stats descriptives + figures
suppressMessages({ library(dplyr); library(lubridate) })
source("src/data_acn.R")

P3_SEUIL   <- 15                             # minutes : départ "anticipé" si > 15 min avant l'heure déclarée
P3_TZ      <- "America/Los_Angeles"
P3_TRAIN_END <- as.POSIXct("2019-09-01", tz = P3_TZ)  # train : sept. 2018 - août 2019
P3_VAL_END   <- as.POSIXct("2019-12-01", tz = P3_TZ)  # validation : sept. - nov. 2019
P3_TEST_END  <- as.POSIXct("2020-03-01", tz = P3_TZ)  # test : déc. 2019 - fév. 2020 (mars 2020 exclu : COVID)
P3_PRIOR_M <- 5                              # poids (en sessions) du prior pour lisser le taux historique

# Variables autorisées : toutes connues à l'arrivée (aucune de disconnectTime/doneChargingTime/kWhDelivered).
P3_FEATURES <- c("site", "arr_hour", "dep_hour", "dur_decl", "weekend", "dow",
                 "kWhRequested", "WhPerMile", "n_past", "past_rate", "past_gap")

build_p3 <- function(d = load_acn(), seuil = P3_SEUIL) {
  x <- d %>%
    filter(site %in% c("caltech", "jpl"),            # office001 : 93 sessions, toutes en 2020 -> pas de train
           !is.na(requestedDeparture), !is.na(disconnectTime), !is.na(userID)) %>%
    mutate(conn  = with_tz(connectionTime, P3_TZ),
           decl  = with_tz(requestedDeparture, P3_TZ),
           disc  = with_tz(disconnectTime, P3_TZ),
           gap   = as.numeric(difftime(disc, decl, units = "mins")),  # < 0 : parti avant l'heure déclarée
           dur_decl = minutesAvailable / 60,
           dur_real = as.numeric(difftime(disc, conn, units = "hours"))) %>%
    filter(dur_decl > 0, dur_decl <= 24, dur_real > 0, dur_real <= 48, conn < P3_TEST_END) %>%
    mutate(early    = as.integer(gap < -seuil),
           arr_hour = hour(conn) + minute(conn) / 60,
           dep_hour = hour(decl) + minute(decl) / 60,
           dow      = factor(wday(conn, week_start = 1), levels = 1:7,
                             labels = c("lun", "mar", "mer", "jeu", "ven", "sam", "dim")),
           weekend  = as.integer(dow %in% c("sam", "dim")),
           site     = factor(site),
           split    = factor(ifelse(conn < P3_TRAIN_END, "train", ifelse(conn < P3_VAL_END, "val", "test")),
                             levels = c("train", "val", "test")))
  p0 <- mean(x$early[x$split == "train"])
  x <- add_history(x, p0)
  x$kWhRequested[is.na(x$kWhRequested)] <- median(x$kWhRequested[x$split == "train"], na.rm = TRUE)
  x$WhPerMile[is.na(x$WhPerMile)]       <- median(x$WhPerMile[x$split == "train"], na.rm = TRUE)
  x %>% arrange(conn) %>% as.data.frame()
}

# Historique de l'usager calculé uniquement sur ses sessions TERMINÉES avant l'arrivée courante
# (disconnect < conn), donc sans fuite : n_past, taux de départ anticipé lissé, écart moyen (h).
add_history <- function(x, p0, m = P3_PRIOR_M) {
  x$n_past <- 0; x$past_k <- 0; x$past_gsum <- 0
  for (idx in split(seq_len(nrow(x)), x$userID)) {
    o <- idx[order(x$disc[idx])]
    n <- findInterval(as.numeric(x$conn[idx]), as.numeric(x$disc[o]), left.open = TRUE)
    ck <- c(0, cumsum(x$early[o])); cg <- c(0, cumsum(x$gap[o]))
    x$n_past[idx] <- n; x$past_k[idx] <- ck[n + 1]; x$past_gsum[idx] <- cg[n + 1]
  }
  x$past_rate <- (x$past_k + m * p0) / (x$n_past + m)
  x$past_gap  <- ifelse(x$n_past > 0, x$past_gsum / pmax(x$n_past, 1) / 60, 0)
  x$past_k <- x$past_gsum <- NULL
  x
}

if (sys.nframe() == 0) {
  suppressMessages(library(ggplot2))
  dir.create("reports/figures", recursive = TRUE, showWarnings = FALSE)
  dir.create("reports/metrics", recursive = TRUE, showWarnings = FALSE)
  d <- load_acn()
  pres <- d %>% group_by(site) %>%
    summarise(sessions = n(), avec_userID = mean(!is.na(userID)), avec_heure_declaree = mean(!is.na(requestedDeparture)))
  print(pres); write.csv(pres, "reports/metrics/p3_presence.csv", row.names = FALSE)
  x <- build_p3(d)
  cat("Sessions retenues :", nrow(x), " usagers :", length(unique(x$userID)), "\n")
  seuils <- data.frame(seuil_min = c(0, 15, 30, 60, 120),
                       taux_anticipe = sapply(c(0, 15, 30, 60, 120), function(s) mean(x$gap < -s)))
  print(seuils); write.csv(seuils, "reports/metrics/p3_seuils.csv", row.names = FALSE)
  cl <- x %>% group_by(split) %>%
    summarise(sessions = n(), debut = format(min(conn), "%Y-%m-%d"), fin = format(max(conn), "%Y-%m-%d"),
              taux_anticipe = mean(early), usagers = n_distinct(userID))
  print(cl); write.csv(cl, "reports/metrics/p3_classes.csv", row.names = FALSE)
  qs <- quantile(x$gap, c(.1, .25, .5, .75, .9)); print(qs)
  g <- ggplot(x %>% filter(abs(gap) < 600), aes(gap)) +
    geom_histogram(binwidth = 10, fill = "grey60") +
    geom_vline(xintercept = -P3_SEUIL, colour = "red") +
    labs(x = "départ réel - départ déclaré (min)", y = "sessions",
         title = "Écart entre départ réel et départ déclaré (|écart| < 10 h)",
         subtitle = "trait rouge : seuil -15 min, à gauche = départ anticipé")
  ggsave("reports/figures/p3_ecart_depart.png", g, width = 7, height = 4, dpi = 120)
  g2 <- ggplot(x %>% mutate(mois = floor_date(conn, "month")) %>% group_by(mois, site) %>%
                 summarise(taux = mean(early), .groups = "drop"), aes(mois, taux, colour = site)) +
    geom_line() + geom_point() + labs(x = NULL, y = "taux de départ anticipé", title = "Taux mensuel de départ anticipé")
  ggsave("reports/figures/p3_taux_mensuel.png", g2, width = 7, height = 3.5, dpi = 120)
}
