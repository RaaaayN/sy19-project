# GAM (mgcv) : splines cycliques sur l'heure, spline sur le jour, splines sur historique et déclarations.
# Lissages choisis par REML ; select = TRUE ajoute une pénalité qui peut annuler un terme (sélection de variables).
# Rscript src/models_gam.R
source("src/features.R")
suppressMessages(library(mgcv))
set.seed(19)
s <- load_sessions()
tr <- s$split == "train"
rhs <- "site + s(hour, bs = 'cc', k = 20) + s(dow, k = 7) + weekend + known + has_hist + has_decl +
        s(h_n, k = 8) + s(h_dur_mean) + s(h_kwh_mean) + s(h_dur_sd) + s(h_kwh_sd) + s(h_hour_mean) +
        s(h_last_dur) + s(h_last_kwh) + s(decl_kwh) + s(decl_dur)"
res <- list(); edf <- list()
for (y in TARGETS) {
  fit <- bam(as.formula(paste(y, "~", rhs)), data = s[tr, ], method = "fREML", select = TRUE,
             knots = list(hour = c(0, 24)), discrete = TRUE)
  res[[y]] <- eval_rows("GAM (splines)", y, s, as.numeric(predict(fit, s)))
  st <- summary(fit)$s.table
  edf[[y]] <- data.frame(target = y, terme = rownames(st), edf = st[, "edf"], p_value = st[, "p-value"])
  png(sprintf("reports/figures/gam_effets_%s.png", y), width = 1400, height = 1000, res = 110)
  plot(fit, pages = 1, shade = TRUE, seWithMean = TRUE, scale = 0)
  dev.off()
  png(sprintf("reports/figures/gam_heure_jour_%s.png", y), width = 1000, height = 420, res = 110)
  par(mfrow = c(1, 2)); plot(fit, select = 1, shade = TRUE, scale = 0, xlab = "Heure d'arrivée", ylab = paste("Effet sur", y))
  plot(fit, select = 2, shade = TRUE, scale = 0, xlab = "Jour (1 = lundi)", ylab = paste("Effet sur", y)); dev.off()
}
save_results(do.call(rbind, res), "gam.csv")
e <- do.call(rbind, edf); rownames(e) <- NULL
write.csv(e, "reports/results/gam_edf.csv", row.names = FALSE); print(e, digits = 3)
