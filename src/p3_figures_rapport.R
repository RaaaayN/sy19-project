# Projet 3 : figures du rapport PDF (reports/rapport_p3.Rmd). Rscript src/p3_figures_rapport.R
# Ne réentraîne que le GAM (pour ses bandes de confiance) ; tout le reste est relu dans
# reports/metrics/ et reports/preds/, produits par les scripts p3_*.R.
suppressMessages({ library(ggplot2); library(dplyr); library(tidyr); library(patchwork); library(mgcv); library(pROC) })
source("src/p3_data.R")
set.seed(19)
out <- "reports/figures/rapport"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
m <- function(f) read.csv(file.path("reports/metrics", f))
save <- function(g, f, w, h) ggsave(file.path(out, f), g, width = w, height = h, dpi = 200, bg = "white")

# Thème et palette communs ---------------------------------------------------------------
theme_set(theme_minimal(base_size = 9) +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold", size = 10),
        plot.subtitle = element_text(colour = "grey35"), legend.position = "bottom",
        plot.tag = element_text(face = "bold")))
COL_EARLY <- "#C0392B"; COL_OK <- "#2E86C1"; COL_GREY <- "grey55"
FAMILLE_COL <- c("Baseline" = "grey55", "Fragile" = "#E67E22", "Linéaire" = "#2E86C1", "Flexible" = "#C0392B")
NOMS <- c(mlp = "MLP (torch)", random_forest = "Forêt aléatoire", gam = "GAM logistique", bagging = "Bagging",
          logistique = "Logistique", svm_lineaire = "SVM linéaire", lda = "LDA", lasso = "Lasso (L1)",
          svm_rbf = "SVM RBF", arbre = "Arbre CART", qda = "QDA", taux_usager = "Taux historique usager",
          majoritaire = "Classe majoritaire")
FAMILLE <- c(mlp = "Flexible", random_forest = "Flexible", gam = "Flexible", bagging = "Flexible",
             logistique = "Linéaire", svm_lineaire = "Linéaire", lda = "Linéaire", lasso = "Linéaire",
             svm_rbf = "Linéaire", arbre = "Fragile", qda = "Fragile", taux_usager = "Baseline", majoritaire = "Baseline")
VARS <- c(arr_hour = "Heure d'arrivée", dep_hour = "Heure de départ déclarée", dur_decl = "Durée déclarée",
          past_rate = "Taux passé de départs anticipés", n_past = "Nb de sessions passées",
          "log1p(n_past)" = "Nb de sessions passées", lnp = "Nb de sessions passées", past_gap = "Écart moyen passé",
          kWhRequested = "Énergie demandée", WhPerMile = "Consommation déclarée", weekend = "Week-end",
          site = "Site", sitejpl = "Site (JPL)")

x <- build_p3()

# Figure 1 : le phénomène et sa dérive -----------------------------------------------------
taux15 <- mean(x$gap < -15)
f1a <- ggplot(x %>% filter(abs(gap) <= 360), aes(gap, fill = gap < -15)) +
  geom_histogram(binwidth = 10, boundary = -15, colour = "white", linewidth = 0.1) +
  geom_vline(xintercept = -15, linetype = "dashed") +
  scale_fill_manual(values = c(`TRUE` = COL_EARLY, `FALSE` = COL_OK),
                    labels = c(`TRUE` = "départ anticipé (> 15 min)", `FALSE` = "à l'heure ou en retard"), name = NULL) +
  scale_x_continuous(breaks = seq(-360, 360, 120), labels = function(b) paste0(b / 60, " h")) +
  annotate("text", x = -200, y = Inf, vjust = 2, size = 3, fontface = "bold",
           label = sprintf("%.1f %% des sessions", 100 * taux15), colour = COL_EARLY) +
  labs(title = "Écart départ réel - départ déclaré", subtitle = "sessions caltech + jpl, |écart| ≤ 6 h",
       x = "écart (négatif = parti avant l'heure annoncée)", y = "sessions", tag = "a")
mens <- x %>% mutate(mois = as.Date(floor_date(conn, "month"))) %>% group_by(mois, split) %>%
  summarise(taux = mean(early), n = n(), .groups = "drop") %>%
  filter(n >= 100)                       # écarte le 31/08/2018 (bord de fuseau horaire, 1 jour)
bandes <- mens %>% group_by(split) %>% summarise(debut = min(mois), fin = max(mois) + 31)
f1b <- ggplot(mens, aes(mois, taux)) +
  geom_rect(data = bandes, aes(xmin = debut, xmax = fin, ymin = -Inf, ymax = Inf, fill = split),
            inherit.aes = FALSE, alpha = 0.18) +
  geom_line(colour = "grey25") + geom_point(aes(size = n), colour = "grey25") +
  scale_fill_manual(values = c(train = "grey60", val = "#F5B041", test = COL_EARLY),
                    labels = c(train = "train", val = "validation", test = "test"), name = NULL) +
  scale_size_area(max_size = 2.5, guide = "none") +
  scale_y_continuous(labels = scales::percent, limits = c(0, NA)) +
  scale_x_date(date_labels = "%m/%y") +
  labs(title = "Taux mensuel de départ anticipé", subtitle = "découpage temporel et dérive de la cible",
       x = NULL, y = NULL, tag = "b")
save(f1a + f1b + plot_layout(widths = c(1.15, 1)), "fig1_probleme.png", 7.2, 2.9)

# Figure 2 : comparaison des modèles -----------------------------------------------------
cmp <- m("p3_comparaison.csv") %>% filter(modele != "majoritaire") %>%
  mutate(nom = factor(NOMS[modele], levels = rev(NOMS[modele])), famille = FAMILLE[modele])
base_auc <- cmp$auc_test[cmp$modele == "taux_usager"]
f2a <- ggplot(cmp, aes(auc_test, nom, colour = famille)) +
  geom_vline(xintercept = base_auc, linetype = "dotted", colour = COL_GREY) +
  geom_errorbar(aes(xmin = auc_test_ic_bas, xmax = auc_test_ic_haut), width = 0, linewidth = 0.8, orientation = "y") +
  geom_point(size = 2.2) + geom_point(aes(x = auc_val), shape = 4, size = 1.8, colour = "black") +
  scale_colour_manual(values = FAMILLE_COL, name = NULL) +
  labs(title = "AUC sur le test (IC 95 % DeLong)", subtitle = "× = AUC de validation (sert au choix du modèle)",
       x = "AUC", y = NULL, tag = "a")
f2b <- ggplot(cmp, aes(brier_test, nom, fill = famille)) +
  geom_col(width = 0.65) +
  geom_text(aes(label = sprintf("%.3f", brier_test)), hjust = -0.15, size = 2.5) +
  scale_fill_manual(values = FAMILLE_COL, guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.25))) +
  labs(title = "Score de Brier (test)", subtitle = "plus bas = probabilités plus justes", x = NULL, y = NULL, tag = "b") +
  theme(axis.text.y = element_blank())
save(f2a + f2b + plot_layout(widths = c(1.6, 1), guides = "collect"), "fig2_comparaison.png", 7.2, 3.3)

# Figure 3 : ROC et calibration ----------------------------------------------------------
sel <- c("mlp", "random_forest", "gam", "logistique", "qda", "taux_usager")
preds <- bind_rows(lapply(sel, function(k) readRDS(sprintf("reports/preds/p3_%s.rds", k)) %>%
                            filter(split == "test") %>% mutate(modele = k)))
COL6 <- setNames(c("#C0392B", "#8E44AD", "#E67E22", "#2E86C1", "#16A085", "grey50"), NOMS[sel])
roc_df <- preds %>% group_by(modele) %>% group_modify(function(d, k) {
  r <- roc(d$y, d$p, quiet = TRUE, direction = "<")
  data.frame(fpr = 1 - r$specificities, tpr = r$sensitivities)
}) %>% ungroup() %>% mutate(nom = factor(NOMS[modele], levels = NOMS[sel]))
f3a <- ggplot(roc_df, aes(fpr, tpr, colour = nom)) +
  geom_abline(linetype = "dotted", colour = COL_GREY) + geom_path(linewidth = 0.6) +
  scale_colour_manual(values = COL6, name = NULL) +
  labs(title = "Courbes ROC (test)", x = "taux de faux positifs", y = "taux de vrais positifs", tag = "a")
cal <- preds %>% group_by(modele) %>% mutate(bin = ntile(p, 10)) %>% group_by(modele, bin) %>%
  summarise(p = mean(p), obs = mean(y), .groups = "drop") %>%
  mutate(nom = factor(NOMS[modele], levels = NOMS[sel]))
cl <- m("p3_classes.csv")
f3b <- ggplot(cal, aes(p, obs, colour = nom)) +
  geom_abline(linetype = "dotted", colour = COL_GREY) + geom_line(linewidth = 0.5) + geom_point(size = 1) +
  scale_colour_manual(values = COL6, guide = "none") + coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  annotate("text", x = 0.98, y = 0.03, hjust = 1, vjust = 0, size = 2.5, colour = "grey30",
           label = sprintf("taux positif %.1f %% (train) → %.1f %% (test)\nau-dessus de la diagonale = risque sous-estimé",
                           100 * cl$taux_anticipe[cl$split == "train"], 100 * cl$taux_anticipe[cl$split == "test"])) +
  labs(title = "Calibration (test, déciles)", x = "probabilité prédite", y = "fréquence observée", tag = "b")
save(f3a + f3b + plot_layout(guides = "collect"), "fig3_roc_calibration.png", 7.2, 3.7)

# Figure 4 : ce que le modèle a appris (GAM, mêmes réglages que src/p3_gam.R) ------------
x$lnp <- log1p(x$n_past)
tr <- x[x$split == "train", ]
gfit <- gam(early ~ site + weekend + s(arr_hour, bs = "cc", k = 20) + s(dep_hour, bs = "cc", k = 20) +
              s(dur_decl, k = 10) + s(past_rate, k = 10) + s(lnp, k = 6) + s(past_gap, k = 6) +
              kWhRequested + WhPerMile,
            family = binomial, data = tr, method = "REML", select = TRUE,
            knots = list(arr_hour = c(0, 24), dep_hour = c(0, 24)))
effet <- function(v, grille, lab, fmt) {
  nd <- tr[rep(1, length(grille)), ]; nd[[v]] <- grille
  pr <- predict(gfit, nd, type = "terms", se.fit = TRUE)
  j <- paste0("s(", v, ")")
  data.frame(var = lab, x = grille, fit = pr$fit[, j], se = pr$se.fit[, j], fmt = fmt)
}
eff <- bind_rows(
  effet("dep_hour", seq(0, 24, 0.1), "Heure de départ déclarée", "h"),
  effet("arr_hour", seq(0, 24, 0.1), "Heure d'arrivée", "h"),
  effet("dur_decl", seq(0.25, 14, 0.1), "Durée déclarée (h)", "n"),
  effet("past_rate", seq(0, 0.9, 0.01), "Taux passé de départs anticipés", "p")) %>%
  mutate(var = factor(var, levels = unique(var)), OR = exp(fit), lo = exp(fit - 2 * se), hi = exp(fit + 2 * se))
f4 <- ggplot(eff, aes(x, OR)) +
  geom_hline(yintercept = 1, colour = COL_GREY) +
  geom_ribbon(aes(ymin = lo, ymax = hi), fill = COL_EARLY, alpha = 0.18) +
  geom_line(colour = COL_EARLY, linewidth = 0.7) +
  facet_wrap(~var, nrow = 1, scales = "free_x") + scale_y_log10() +
  scale_x_continuous(breaks = function(l) if (diff(l) > 15) seq(0, 24, 6) else scales::extended_breaks()(l)) +
  labs(title = "Effets partiels du GAM : facteur multiplicatif sur la cote du départ anticipé",
       subtitle = "> 1 : plus de risque ; bande = ± 2 erreurs-types ; échelle logarithmique",
       x = NULL, y = "odds ratio (échelle log)")
save(f4, "fig4_gam_effets.png", 7.2, 2.6)

# Figure 5 : quelles variables comptent ? -------------------------------------------------
las <- m("p3_lasso_coefs.csv") %>% filter(variable != "(Intercept)") %>%
  mutate(nom = VARS[variable], garde = coef_lambda_1se != 0)
f5a <- ggplot(las, aes(coef_lambda_1se, reorder(nom, abs(coef_lambda_min)), fill = garde)) +
  geom_col(width = 0.65) + geom_vline(xintercept = 0, colour = "grey40") +
  geom_point(data = function(d) d[!d$garde, ], aes(x = 0), shape = 4, colour = "grey50", size = 2) +
  geom_text(data = function(d) d[!d$garde, ], aes(x = 0.02, label = "éliminée"), hjust = 0, size = 2.4, colour = "grey50") +
  scale_fill_manual(values = c(`TRUE` = COL_OK, `FALSE` = "grey80"), guide = "none") +
  labs(title = "Lasso (λ 1-SE) : 6 variables sur 10", subtitle = "coefficients sur variables standardisées",
       x = "coefficient", y = NULL, tag = "a")
imp <- m("p3_rf_importance.csv") %>% mutate(nom = VARS[variable])
f5b <- ggplot(imp, aes(MeanDecreaseAccuracy, reorder(nom, MeanDecreaseAccuracy))) +
  geom_segment(aes(x = 0, xend = MeanDecreaseAccuracy, yend = reorder(nom, MeanDecreaseAccuracy)), colour = "grey70") +
  geom_point(colour = "#8E44AD", size = 2.2) +
  labs(title = "Forêt aléatoire : importance", subtitle = "baisse de précision (permutation)", x = NULL, y = NULL, tag = "b")
save(f5a + f5b, "fig5_variables.png", 7.2, 2.8)

# Figure 6 : investigation (ancienneté) --------------------------------------------------
anc <- m("p3_invest_anciennete.csv") %>% mutate(anc = factor(anc, levels = anc))
rang <- m("p3_invest_rang_intra_usager.csv") %>% mutate(rang = factor(rang, levels = rang))
f6a <- ggplot(anc, aes(anc, taux_anticipe)) +
  geom_col(fill = COL_EARLY, width = 0.65, alpha = 0.85) +
  geom_text(aes(label = sprintf("%.1f %%", 100 * taux_anticipe)), vjust = -0.4, size = 2.6) +
  scale_y_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Entre usagers", subtitle = "taux selon le nb de sessions passées", x = "sessions passées", y = NULL, tag = "a")
f6b <- ggplot(rang, aes(rang, taux_anticipe, group = 1)) +
  geom_line(colour = COL_EARLY) + geom_point(colour = COL_EARLY, size = 2) +
  geom_text(aes(label = sprintf("%.1f %%", 100 * taux_anticipe)), vjust = -1, size = 2.6) +
  scale_y_continuous(labels = scales::percent, limits = c(0.2, 0.34)) +
  labs(title = "Chez un même usager", subtitle = "usagers ≥ 40 sessions, rang de la session", x = "rang", y = NULL, tag = "b")
aucl <- anc %>% select(anc, `MLP (torch)` = auc_modele, `Taux historique usager` = auc_taux_usager) %>%
  pivot_longer(-anc, names_to = "score", values_to = "auc") %>% filter(!is.na(auc))
f6c <- ggplot(aucl, aes(anc, auc, colour = score, group = score)) +
  geom_line() + geom_point(size = 2) +
  scale_colour_manual(values = c(`MLP (torch)` = COL_EARLY, `Taux historique usager` = COL_GREY), guide = "none") +
  annotate("text", x = 1.1, y = 0.84, label = "MLP (torch)", colour = COL_EARLY, size = 2.7, hjust = 0) +
  annotate("text", x = 3, y = 0.70, label = "taux historique", colour = "grey40", size = 2.7, hjust = 0.5) +
  labs(title = "Apport du modèle", subtitle = "AUC test par ancienneté", x = "sessions passées", y = "AUC", tag = "c")
save(f6a + f6b + f6c, "fig6_anciennete.png", 7.2, 2.7)

# Figure 7 : usage dans IA04 (marge LLF) -------------------------------------------------
gj <- jsonlite::fromJSON("reports/export/depart_risk_gam.json")
mlp <- readRDS("reports/preds/p3_mlp.rds") %>% filter(split == "test") %>%
  mutate(classe = ifelse(y == 1, "parti > 15 min en avance", "parti à l'heure"))
seuil <- m("p3_comparaison.csv") %>% filter(modele == "mlp") %>% pull(seuil)
f7 <- ggplot(mlp, aes(p, fill = classe)) +
  geom_density(alpha = 0.55, colour = NA) +
  geom_vline(xintercept = seuil, linetype = "dashed") +
  annotate("text", x = seuil, y = Inf, vjust = 1.5, hjust = -0.05, size = 2.6, label = paste("seuil F1 =", formatC(seuil, format = "f", digits = 2, decimal.mark = ","))) +
  scale_fill_manual(values = c(`parti > 15 min en avance` = COL_EARLY, `parti à l'heure` = COL_OK), name = NULL) +
  scale_x_continuous(sec.axis = sec_axis(~ . * gj$avance_mediane_si_anticipe_min,
                                         name = "marge LLF = p × avance médiane (min)")) +
  labs(title = "Probabilité prédite (MLP, test) et marge de sécurité pour le LLF", x = "probabilité de départ anticipé", y = "densité")
save(f7, "fig7_marge_llf.png", 7.2, 2.5)
cat("Figures écrites dans", out, "\n")
