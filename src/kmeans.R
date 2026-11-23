# Étape 3 : baseline k-means, choix de k (coude, silhouette). Rscript src/kmeans.R
suppressPackageStartupMessages({ library(ggplot2); library(patchwork); library(cluster); library(mclust) })
source("src/features.R")
L <- load_users()
nearest <- function(z, centers) apply(z, 1, function(x) which.min(colSums((t(centers) - x)^2)))
sil <- function(cl, z) if (length(unique(cl)) < 2) NA else mean(silhouette(cl, dist(z))[, 3])

set.seed(SEED)
res <- do.call(rbind, lapply(2:10, function(k) {
  km <- kmeans(L$ztr, k, nstart = 50, iter.max = 100)
  cl_te <- nearest(L$zte, km$centers)
  km_te <- kmeans(L$zte, k, nstart = 50, iter.max = 100)   # k-means refait sur le test seul
  data.frame(k = k, inertie = km$tot.withinss, sil_train = sil(km$cluster, L$ztr),
             sil_test = sil(cl_te, L$zte), ari_test = adjustedRandIndex(cl_te, km_te$cluster))
}))
print(round(res, 3)); write.csv(round(res, 4), "reports/results/kmeans_choix_k.csv", row.names = FALSE)

g1 <- ggplot(res, aes(k, inertie)) + geom_line() + geom_point() + labs(title = "Coude (inertie intra, train)") + theme_minimal()
g2 <- ggplot(tidyr::pivot_longer(res, c(sil_train, sil_test, ari_test)), aes(k, value, colour = name)) +
  geom_line() + geom_point() + labs(title = "Silhouette et ARI de généralisation", y = NULL, colour = NULL) + theme_minimal()
ggsave("reports/figures/kmeans_choix_k.png", g1 + g2, width = 11, height = 4, dpi = 110)

# k retenu : meilleure silhouette train parmi k >= 3 (k = 2 oppose seulement arrivées du matin et de l après-midi, trop grossier pour des types d agents)
K_KM <- res$k[res$k >= 3][which.max(res$sil_train[res$k >= 3])]
set.seed(SEED)
km <- kmeans(L$ztr, K_KM, nstart = 50, iter.max = 100)
cat("k retenu :", K_KM, "| tailles :", km$size, "\n")
saveRDS(list(k = K_KM, km = km, cl_test = nearest(L$zte, km$centers)), "reports/results/kmeans.rds")
