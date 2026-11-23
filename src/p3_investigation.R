# Projet 3, étape 10b : investigation originale. Rscript src/p3_investigation.R (après p3_compare.R)
# A) La fiabilité de l'heure déclarée évolue-t-elle avec l'ancienneté de l'usager, et le modèle
#    apporte-t-il plus que l'historique pour les nouveaux usagers ?
# B) Effet du rééquilibrage des classes (pondération, sous-échantillonnage) sur AUC, F1 et calibration.
suppressMessages({ library(ggplot2); library(dplyr); library(randomForest) })
source("src/p3_eval.R")
set.seed(19)
x <- build_p3(); m <- p3_matrix(x)
best <- sub("modele_retenu,", "", readLines("reports/metrics/p3_selection.csv")[1])

# ---- A) ancienneté = nombre de sessions passées terminées (n_past) ----
x$anc <- cut(x$n_past, c(-1, 0, 4, 19, 49, Inf), labels = c("0", "1-4", "5-19", "20-49", "50+"))
a1 <- x %>% group_by(anc) %>% summarise(sessions = n(), taux_anticipe = mean(early),
                                        ecart_abs_median_min = median(abs(gap)), .groups = "drop")
pb <- readRDS(sprintf("reports/preds/p3_%s.rds", best)); pu <- readRDS("reports/preds/p3_taux_usager.rds")
te <- x[x$split == "test", c("sessionID", "anc")] %>%
  inner_join(pb %>% select(sessionID, y, p_best = p), by = "sessionID") %>%
  inner_join(pu %>% select(sessionID, p_usager = p), by = "sessionID")
a2 <- te %>% group_by(anc) %>% summarise(sessions_test = n(), taux_test = mean(y),
  auc_modele = if (length(unique(y)) == 2) p3_auc(y, p_best) else NA,
  auc_taux_usager = if (length(unique(y)) == 2 && sd(p_usager) > 0) p3_auc(y, p_usager) else NA, .groups = "drop")
inv_a <- left_join(a1, a2, by = "anc"); print(inv_a, width = 200)
write.csv(inv_a, "reports/metrics/p3_invest_anciennete.csv", row.names = FALSE)

# Évolution intra-usager : rang de la session chez l'usager (usagers avec >= 40 sessions).
r <- x %>% group_by(userID) %>% filter(n() >= 40) %>% arrange(conn, .by_group = TRUE) %>%
  mutate(rang = cut(row_number(), c(0, 5, 10, 20, 40, Inf), labels = c("1-5", "6-10", "11-20", "21-40", "41+"))) %>%
  group_by(rang) %>% summarise(usagers = n_distinct(userID), sessions = n(), taux_anticipe = mean(early), .groups = "drop")
print(r); write.csv(r, "reports/metrics/p3_invest_rang_intra_usager.csv", row.names = FALSE)
g <- ggplot(te %>% tidyr::pivot_longer(c(p_best, p_usager), names_to = "score") %>%
              group_by(anc, score) %>% summarise(auc = p3_auc(y, value), .groups = "drop") %>%
              mutate(score = ifelse(score == "p_best", best, "taux_usager")),
            aes(anc, auc, colour = score, group = score)) + geom_line() + geom_point() +
  labs(x = "sessions passées de l'usager", y = "AUC (test)", title = "Apport du modèle selon l'ancienneté de l'usager")
ggsave("reports/figures/p3_invest_anciennete.png", g, width = 6.5, height = 4, dpi = 120)

# ---- B) rééquilibrage des classes ----
dtr <- data.frame(m$X[m$tr, ], early = m$y[m$tr]); dall <- data.frame(m$X)
w <- ifelse(dtr$early == 1, 0.5 / mean(dtr$early), 0.5 / mean(1 - dtr$early))
i0 <- which(dtr$early == 0); i1 <- which(dtr$early == 1); us <- c(i1, sample(i0, length(i1)))
lg <- function(data, weights = NULL)
  suppressWarnings(predict(glm(early ~ ., binomial, data = data, weights = weights), dall, type = "response"))
vars <- c("site", "arr_hour", "dep_hour", "dur_decl", "weekend", "kWhRequested", "WhPerMile", "n_past", "past_rate", "past_gap")
rf_tr <- x[m$tr, vars]; yf <- factor(m$y[m$tr]); n1 <- sum(m$y[m$tr])
rfp <- function(...) { set.seed(19); predict(randomForest(rf_tr, yf, ntree = 300, mtry = 2, ...), x[, vars], type = "prob")[, "1"] }
variants <- list(
  logistique_aucun   = lg(dtr),
  logistique_poids   = lg(dtr, w),
  logistique_sous_ech = lg(dtr[us, ]),
  rf_aucun           = rfp(),
  rf_sous_ech        = rfp(strata = yf, sampsize = c(n1, n1)))
inv_b <- do.call(rbind, lapply(names(variants), function(n) {
  p <- variants[[n]]; v <- m$val; t <- m$te
  thr <- p3_best_thr(m$y[v], p[v]); mt <- p3_metrics(m$y[t], p[t], thr)
  data.frame(variante = n, auc_test = mt$auc, f1_test = mt$f1, precision = mt$precision, recall = mt$recall,
             f1_test_seuil_05 = p3_metrics(m$y[t], p[t], 0.5)$f1, brier_test = mt$brier,
             proba_moyenne_test = mean(p[t]), taux_observe_test = mean(m$y[t]), seuil_val = thr)
}))
print(inv_b, digits = 3); write.csv(inv_b, "reports/metrics/p3_invest_reequilibrage.csv", row.names = FALSE)
