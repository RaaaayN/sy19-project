# Tableau comparatif des trois méthodes de clustering. Rscript src/comparaison.R
suppressPackageStartupMessages({ library(cluster); library(mclust) })
source("src/features.R"); source("src/em_gmm.R")
L <- load_users()
km <- readRDS("reports/results/kmeans.rds"); kk <- readRDS("reports/results/kkmeans.rds")
pr <- readRDS("reports/results/profils.rds"); va <- readRDS("reports/results/validation.rds")
gm_tr <- predict_gmm(pr$fit, L$ztr)$cluster; gm_te <- predict_gmm(pr$fit, L$zte)$cluster
sil <- function(cl, z) mean(silhouette(cl, dist(z))[, 3])
st <- setNames(va$stab$ari_moyen, va$stab$methode)
tab <- data.frame(
  methode = c("k-means (baseline)", "GMM, EM maison", "kernel k-means"),
  K = c(km$k, nrow(pr$fit$mu), kk$k),
  critere_choix = c("coude + silhouette", "BIC (= max. vraisemblance test)", "silhouette sur grille sigma x k"),
  sil_train = c(sil(km$km$cluster, L$ztr), sil(gm_tr, L$ztr), sil(kk$cl, L$ztr)),
  sil_test = c(sil(km$cl_test, L$zte), sil(gm_te, L$zte), sil(kk$cl_test, L$zte)),
  ari_bootstrap = st[c("kmeans", "gmm", "kkmeans")],
  ari_avec_gmm = c(adjustedRandIndex(km$km$cluster, gm_tr), 1, adjustedRandIndex(kk$cl, gm_tr)))
tab[, 4:7] <- round(tab[, 4:7], 3)
print(tab, row.names = FALSE); write.csv(tab, "reports/results/comparaison.csv", row.names = FALSE)
