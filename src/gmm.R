# Étape 4 : GMM par EM maison vs mclust, choix de K par BIC/AIC. Rscript src/gmm.R
suppressPackageStartupMessages({ library(ggplot2); library(mclust) })
source("src/features.R"); source("src/em_gmm.R")
L <- load_users()
Ks <- 1:7
t0 <- Sys.time()
fits <- lapply(Ks, function(K) em_gmm(L$ztr, K, n_init = 50))
# K pour lequel tous les essais ont produit une composante de moins de p + 1 usagers : écarté (NA)
get <- function(f, x) if (is.null(f)) NA else f[[x]]
cat("EM maison :", round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), "s\n")
mc <- Mclust(L$ztr, G = Ks, modelNames = "VVV", verbose = FALSE)
res <- data.frame(K = Ks,
  npar = gmm_npar(Ks, ncol(L$ztr)),
  loglik = sapply(fits, get, "loglik"),
  BIC = sapply(fits, get, "bic"), AIC = sapply(fits, get, "aic"),
  ll_test_par_usager = sapply(fits, function(f) if (is.null(f)) NA else predict_gmm(f, L$zte)$loglik / nrow(L$zte)),
  BIC_mclust = -mc$BIC[as.character(Ks), "VVV"])        # mclust : BIC = 2 logL - p log n (signe opposé)
print(round(res, 2)); write.csv(round(res, 3), "reports/results/gmm_choix_K.csv", row.names = FALSE)

K_BIC <- res$K[which.min(res$BIC)]; K_AIC <- res$K[which.min(res$AIC)]
K_TEST <- res$K[which.max(res$ll_test_par_usager)]
cat("K par BIC :", K_BIC, "| AIC :", K_AIC, "| vraisemblance test :", K_TEST, "\n")

mc3 <- Mclust(L$ztr, G = K_BIC, modelNames = "VVV", verbose = FALSE)
ari_mc <- adjustedRandIndex(fits[[K_BIC]]$cluster, mc3$classification)
cat("ARI EM maison vs mclust (VVV, K =", K_BIC, ") :", round(ari_mc, 3),
    "| logL maison", round(fits[[K_BIC]]$loglik, 2), "vs mclust", round(mc3$loglik, 2), "\n")
# Modèle complet mclust (14 paramétrisations de covariance) pour référence
mc_all <- Mclust(L$ztr, G = Ks, verbose = FALSE)
cat("mclust toutes covariances :", mc_all$modelName, "G =", mc_all$G, "\n")

g <- ggplot(tidyr::pivot_longer(res[, c("K", "BIC", "AIC", "BIC_mclust")], -K), aes(K, value, colour = name)) +
  geom_line(na.rm = TRUE) + geom_point(na.rm = TRUE) + labs(title = "GMM covariances pleines : BIC et AIC (plus bas = meilleur)", y = NULL, colour = NULL) +
  theme_minimal()
ggsave("reports/figures/gmm_choix_K.png", g, width = 7, height = 4, dpi = 110)
saveRDS(list(res = res, fits = fits, K_BIC = K_BIC, K_AIC = K_AIC, K_TEST = K_TEST, ari_mclust = ari_mc,
             mclust_vvv = mc, mclust_all = list(model = mc_all$modelName, G = mc_all$G, cl = mc_all$classification)),
        "reports/results/gmm.rds")
