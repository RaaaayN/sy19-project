# Investigation : apport des déclarations de l'usager par rapport à l'historique seul,
# selon la taille d'historique disponible. Modèle : GAM (rapide, donc ré-ajustable pour chaque jeu de variables).
# Rscript src/investigation.R
source("src/features.R")
suppressMessages({ library(mgcv); library(ggplot2) })
set.seed(19)
s <- load_sessions()
tr <- s$split == "train"
lin <- c("site", "weekend", "known", "has_hist", "has_decl")
term <- function(v) if (v %in% lin) v else if (v == "hour") "s(hour, bs = 'cc', k = 20)" else if (v == "dow") "s(dow, k = 7)" else if (v == "h_n") "s(h_n, k = 8)" else sprintf("s(%s)", v)
sets <- list("Contexte seul"            = BASE_VARS,
             "Contexte + historique"    = c(BASE_VARS, "has_hist", HIST_VARS),
             "Contexte + déclarations"  = c(BASE_VARS, DECL_VARS),
             "Contexte + hist. + décl." = ALL_VARS)
n_hist <- round(expm1(s$h_n))
s$bucket <- cut(n_hist, c(-1, 0, 4, 19, 49, Inf), labels = c("0 (inconnu ou 1re)", "1-4", "5-19", "20-49", "50+"))
te <- s$split == "test"
rows <- list()
for (y in TARGETS) for (nm in names(sets)) {
  f <- as.formula(paste(y, "~", paste(sapply(sets[[nm]], term), collapse = " + ")))
  p <- as.numeric(predict(bam(f, data = s[tr, ], method = "fREML", discrete = TRUE, knots = list(hour = c(0, 24))), s))
  for (b in c("Tous", levels(s$bucket))) {
    i <- te & (b == "Tous" | s$bucket == b)
    rows[[length(rows) + 1]] <- data.frame(target = y, variables = nm, historique = b, n = sum(i), t(metrics(s[[y]][i], p[i])))
  }
}
r <- do.call(rbind, rows)
save_results(r, "investigation.csv")
r$variables <- factor(r$variables, levels = names(sets)); r$historique <- factor(r$historique, levels = c("Tous", levels(s$bucket)))
ggsave("reports/figures/investigation.png",
  ggplot(r, aes(historique, MAE, fill = variables)) + geom_col(position = "dodge") +
    facet_wrap(~target, scales = "free_y", labeller = as_labeller(c(duration = "Durée (h)", energy = "Énergie (kWh)"))) +
    labs(x = "Nombre de sessions passées de l'usager", y = "MAE sur le test", fill = NULL) +
    theme_minimal() + theme(legend.position = "bottom", axis.text.x = element_text(angle = 20, hjust = 1)),
  width = 10, height = 5, dpi = 120)
