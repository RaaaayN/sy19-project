# Projet 3, étape 6 : GAM logistique, splines sur les heures (mgcv). Rscript src/p3_gam.R
suppressMessages(library(mgcv))
source("src/p3_eval.R")
x <- build_p3(); x$lnp <- log1p(x$n_past)
tr <- x[x$split == "train", ]
# Splines cycliques (bs = "cc") pour les heures (23h59 voisin de 0h), splines de régression
# pénalisées pour la durée déclarée et l'historique ; lissage choisi par REML sur le train,
# select = TRUE permet de mettre à zéro un terme inutile (sélection de variables).
fit <- gam(early ~ site + weekend + s(arr_hour, bs = "cc", k = 20) + s(dep_hour, bs = "cc", k = 20) +
             s(dur_decl, k = 10) + s(past_rate, k = 10) + s(lnp, k = 6) + s(past_gap, k = 6) +
             kWhRequested + WhPerMile,
           family = binomial, data = tr, method = "REML", select = TRUE,
           knots = list(arr_hour = c(0, 24), dep_hour = c(0, 24)))
print(summary(fit))
s <- summary(fit)$s.table
write.csv(data.frame(terme = rownames(s), s, check.names = FALSE), "reports/metrics/p3_gam_termes.csv", row.names = FALSE)
png("reports/figures/p3_gam_effets.png", width = 1000, height = 650)
plot(fit, pages = 1, shade = TRUE, scale = 0, seWithMean = TRUE); dev.off()
p3_save("gam", x, as.numeric(predict(fit, x, type = "response")))
