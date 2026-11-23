# Projet 3, étape 10a : comparaison des modèles (AUC, F1, confusion, ROC, calibration). Rscript src/p3_compare.R
suppressMessages({ library(ggplot2); library(dplyr); library(pROC) })
source("src/p3_eval.R")
mods <- c("majoritaire", "taux_usager", "logistique", "lasso", "lda", "qda", "gam", "arbre",
          "bagging", "random_forest", "svm_lineaire", "svm_rbf", "mlp")
met <- do.call(rbind, lapply(mods, function(n) read.csv(sprintf("reports/metrics/p3_models/%s.csv", n))[, 1:14]))
preds <- lapply(setNames(mods, mods), function(n) readRDS(sprintf("reports/preds/p3_%s.rds", n)))

# IC 95 % de l'AUC test (DeLong).
ci <- t(sapply(mods, function(n) { t <- preds[[n]][preds[[n]]$split == "test", ]
  as.numeric(ci.auc(roc(t$y, t$p, quiet = TRUE, direction = "<"), method = "delong"))[c(1, 3)] }))
tab <- met %>% select(modele, jeu, auc, f1, precision, recall, accuracy, brier, seuil) %>%
  tidyr::pivot_wider(names_from = jeu, values_from = c(auc, f1, precision, recall, accuracy, brier)) %>%
  mutate(auc_test_ic_bas = ci[modele, 1], auc_test_ic_haut = ci[modele, 2]) %>%
  arrange(desc(auc_val)) %>% as.data.frame()
write.csv(tab, "reports/metrics/p3_comparaison.csv", row.names = FALSE)
print(tab[, c("modele", "auc_val", "auc_test", "auc_test_ic_bas", "auc_test_ic_haut", "f1_val", "f1_test", "brier_test")], digits = 3)

# Sélection : meilleur AUC de validation (le test ne sert qu'à rapporter).
best <- tab$modele[1]; cat("Modèle retenu (AUC validation) :", best, "\n")
bt <- preds[[best]][preds[[best]]$split == "test", ]; bu <- preds$taux_usager[preds$taux_usager$split == "test", ]
rt <- roc.test(roc(bt$y, bt$p, quiet = TRUE, direction = "<"), roc(bu$y, bu$p, quiet = TRUE, direction = "<"), method = "delong")
cat("DeLong", best, "vs taux_usager (test) : p =", format.pval(rt$p.value), "\n")
writeLines(c(sprintf("modele_retenu,%s", best), sprintf("delong_p_vs_taux_usager,%g", rt$p.value)),
           "reports/metrics/p3_selection.csv")

# Matrices de confusion test (seuil choisi sur la validation).
conf <- met %>% filter(jeu == "test") %>% select(modele, seuil, tp, fp, fn, tn)
write.csv(conf, "reports/metrics/p3_confusion_test.csv", row.names = FALSE)

show <- c("taux_usager", "logistique", "gam", "random_forest", "svm_rbf", "mlp", "qda")
te <- bind_rows(lapply(show, function(n) cbind(preds[[n]][preds[[n]]$split == "test", ], modele = n)))
roc_df <- te %>% group_by(modele) %>% group_modify(function(g, k) {
  r <- roc(g$y, g$p, quiet = TRUE, direction = "<"); data.frame(fpr = 1 - r$specificities, tpr = r$sensitivities) })
g1 <- ggplot(roc_df, aes(fpr, tpr, colour = modele)) + geom_line() + geom_abline(lty = 2, colour = "grey50") +
  coord_equal() + labs(x = "taux de faux positifs", y = "taux de vrais positifs", title = "Courbes ROC (test)")
ggsave("reports/figures/p3_roc_test.png", g1, width = 6.5, height = 5.5, dpi = 120)

cal <- te %>% group_by(modele) %>% mutate(bin = ntile(p, 10)) %>% group_by(modele, bin) %>%
  summarise(p_moy = mean(p), obs = mean(y), .groups = "drop")
g2 <- ggplot(cal, aes(p_moy, obs, colour = modele)) + geom_abline(lty = 2, colour = "grey50") +
  geom_line() + geom_point() + coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "probabilité prédite (décile)", y = "fréquence observée", title = "Calibration (test, déciles)")
ggsave("reports/figures/p3_calibration_test.png", g2, width = 6.5, height = 5.5, dpi = 120)

g3 <- ggplot(tab %>% filter(modele != "majoritaire"), aes(reorder(modele, auc_test), auc_test)) +
  geom_pointrange(aes(ymin = auc_test_ic_bas, ymax = auc_test_ic_haut)) + geom_point(aes(y = auc_val), shape = 4, colour = "red") +
  coord_flip() + labs(x = NULL, y = "AUC", title = "AUC test (IC 95 % DeLong) ; croix rouge = AUC validation")
ggsave("reports/figures/p3_auc_modeles.png", g3, width = 7, height = 4.5, dpi = 120)
