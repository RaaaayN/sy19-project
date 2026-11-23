# Étape 5 : kernel k-means (noyau gaussien), grille (sigma, k). Rscript src/kkmeans.R
suppressPackageStartupMessages({ library(kernlab); library(cluster); library(mclust); library(ggplot2) })
source("src/features.R")
sil <- function(cl, z) if (length(unique(cl)) < 2) NA else mean(silhouette(cl, dist(z))[, 3])

# Affectation de nouveaux points : distance au barycentre de chaque cluster dans l'espace de redescription
kk_assign <- function(znew, ztr, cl, kern) {
  Kx <- kernelMatrix(kern, znew, ztr)
  d <- sapply(sort(unique(cl)), function(c) {
    i <- cl == c
    1 - 2 * rowMeans(Kx[, i, drop = FALSE]) + mean(kernelMatrix(kern, ztr[i, , drop = FALSE]))
  })
  max.col(-d, ties.method = "first")
}
# Meilleur de plusieurs départs (kkmeans n'a pas de nstart) : inertie dans l'espace du noyau minimale
kk_fit <- function(z, k, sigma, n_start = 10) {
  fits <- lapply(seq_len(n_start), function(i) tryCatch(
    kkmeans(z, centers = k, kernel = "rbfdot", kpar = list(sigma = sigma)), error = function(e) NULL))
  fits <- Filter(function(f) !is.null(f) && length(unique(f@.Data)) == k, fits)
  if (!length(fits)) return(NULL)
  fits[[which.min(sapply(fits, function(f) sum(withinss(f))))]]
}

if (sys.nframe() == 0) {
  L <- load_users()
  set.seed(SEED)
  s_med <- sigest(L$ztr, scaled = FALSE)[2]          # heuristique de la médiane
  sigmas <- signif(s_med * c(0.25, 0.5, 1, 2), 3)
  res <- do.call(rbind, lapply(sigmas, function(sg) do.call(rbind, lapply(2:8, function(k) {
    f <- kk_fit(L$ztr, k, sg)
    if (is.null(f)) return(data.frame(sigma = sg, k = k, sil_train = NA, sil_test = NA))
    cl_te <- kk_assign(L$zte, L$ztr, f@.Data, rbfdot(sg))
    data.frame(sigma = sg, k = k, sil_train = sil(f@.Data, L$ztr), sil_test = sil(cl_te, L$zte))
  }))))
  print(res, digits = 3); write.csv(res, "reports/results/kkmeans_grille.csv", row.names = FALSE)

  # Même règle que pour k-means : meilleure silhouette train avec k >= 3
  r3 <- res[res$k >= 3 & !is.na(res$sil_train), ]
  best <- r3[which.max(r3$sil_train), ]
  set.seed(SEED)
  f <- kk_fit(L$ztr, best$k, best$sigma)
  cl_te <- kk_assign(L$zte, L$ztr, f@.Data, rbfdot(best$sigma))
  cat("Retenu : sigma =", best$sigma, "k =", best$k, "| tailles :", table(f@.Data), "\n")
  km <- readRDS("reports/results/kmeans.rds"); gm <- readRDS("reports/results/gmm.rds")
  cat("ARI avec k-means :", round(adjustedRandIndex(f@.Data, km$km$cluster), 3),
      "| avec GMM :", round(adjustedRandIndex(f@.Data, gm$fits[[gm$K_BIC]]$cluster), 3), "\n")
  g <- ggplot(res, aes(k, sil_train, colour = factor(sigma))) + geom_line(na.rm = TRUE) + geom_point(na.rm = TRUE) +
    labs(title = "Kernel k-means : silhouette (train) selon sigma et k", colour = "sigma") + theme_minimal()
  ggsave("reports/figures/kkmeans_grille.png", g, width = 7, height = 4, dpi = 110)
  saveRDS(list(sigma = best$sigma, k = best$k, cl = f@.Data, cl_test = cl_te), "reports/results/kkmeans.rds")
}
