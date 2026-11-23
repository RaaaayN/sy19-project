# Étape 7 : stabilité par bootstrap (ARI) et explication des profils par LDA et arbre. Rscript src/validation.R
suppressPackageStartupMessages({ library(mclust); library(MASS); library(rpart); library(kernlab); library(ggplot2) })
source("src/features.R"); source("src/em_gmm.R")
source("src/kkmeans.R")                           # kk_fit, kk_assign (l'analyse ne tourne qu'en Rscript)
L <- load_users()
km <- readRDS("reports/results/kmeans.rds"); kk <- readRDS("reports/results/kkmeans.rds")
pr <- readRDS("reports/results/profils.rds"); gm_ref <- predict_gmm(pr$fit, L$ztr)$cluster
nearest <- function(z, centers) apply(z, 1, function(x) which.min(colSums((t(centers) - x)^2)))

# Bootstrap : modèle réappris sur un rééchantillon des usagers du train, puis appliqué à tout le train
B <- 50; set.seed(SEED)
boot <- do.call(rbind, lapply(seq_len(B), function(b) {
  i <- sample(nrow(L$ztr), replace = TRUE); zb <- L$ztr[i, ]
  a_km <- adjustedRandIndex(nearest(L$ztr, kmeans(zb, km$k, nstart = 20)$centers), km$km$cluster)
  fb <- em_gmm(zb + rnorm(length(zb), sd = 1e-3), nrow(pr$fit$mu), n_init = 10, seed = b)  # léger bruit : doublons
  a_gm <- if (is.null(fb)) NA else adjustedRandIndex(predict_gmm(fb, L$ztr)$cluster, gm_ref)
  a_kk <- NA
  if (b <= 25) {                                   # kernel k-means plus coûteux : 25 rééchantillons
    zu <- unique(zb); f <- kk_fit(zu, kk$k, kk$sigma, n_start = 3)
    if (!is.null(f)) a_kk <- adjustedRandIndex(kk_assign(L$ztr, zu, f@.Data, rbfdot(kk$sigma)), kk$cl)
  }
  data.frame(b = b, kmeans = a_km, gmm = a_gm, kkmeans = a_kk)
}))
st <- data.frame(methode = c("kmeans", "gmm", "kkmeans"),
                 ari_moyen = sapply(boot[, -1], mean, na.rm = TRUE), ari_sd = sapply(boot[, -1], sd, na.rm = TRUE),
                 B = sapply(boot[, -1], function(x) sum(!is.na(x))))
print(st, digits = 3); write.csv(st, "reports/results/stabilite_bootstrap.csv", row.names = FALSE)
bl <- tidyr::pivot_longer(boot, -b, values_drop_na = TRUE)
ggsave("reports/figures/stabilite_bootstrap.png",
       ggplot(bl, aes(name, value)) + geom_boxplot() + geom_jitter(width = .1, alpha = .4) +
         labs(x = NULL, y = "ARI avec la partition de référence", title = "Stabilité bootstrap des partitions (train)") +
         theme_minimal(), width = 6, height = 4, dpi = 110)

# Explication : LDA et arbre prédisent le profil GMM à partir des variables brutes (interprétables)
tr <- L$tr; te <- L$te
tr$profil <- factor(pr$noms[gm_ref], pr$noms); te$profil <- factor(pr$noms[predict_gmm(pr$fit, L$zte)$cluster], pr$noms)
fml <- reformulate(CLUST_VARS, "profil")
lda_fit <- lda(fml, data = tr)
set.seed(SEED)
tree <- rpart(fml, data = tr, method = "class", control = rpart.control(cp = 0.001, xval = 10))
cp_best <- tree$cptable[which.min(tree$cptable[, "xerror"]), "CP"]   # élagage par validation croisée
tree <- prune(tree, cp = cp_best)
ex <- data.frame(modele = c("LDA", "arbre"),
  acc_train = c(mean(predict(lda_fit, tr)$class == tr$profil), mean(predict(tree, tr, type = "class") == tr$profil)),
  acc_test = c(mean(predict(lda_fit, te)$class == te$profil), mean(predict(tree, te, type = "class") == te$profil)),
  feuilles = c(NA, sum(tree$frame$var == "<leaf>")))
print(ex, digits = 3); write.csv(ex, "reports/results/explication_profils.csv", row.names = FALSE)
cat("Coefficients LDA (variables brutes) :\n"); print(round(lda_fit$scaling, 3))
cat("Importance des variables (arbre) :\n"); print(round(tree$variable.importance, 1))
print(tree)
png("reports/figures/arbre_profils.png", width = 1000, height = 650, res = 110)
plot(tree, uniform = TRUE, margin = 0.08); text(tree, use.n = TRUE, cex = 0.7, all = FALSE)
title("Arbre de décision qui prédit le profil GMM")
dev.off()
saveRDS(list(boot = boot, stab = st, lda = lda_fit, tree = tree, explication = ex), "reports/results/validation.rds")
