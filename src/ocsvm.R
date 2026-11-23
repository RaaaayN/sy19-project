# Étape 8 : one-class SVM pour repérer les usagers atypiques. Rscript src/ocsvm.R
suppressPackageStartupMessages({ library(e1071); library(ggplot2); library(dplyr) })
source("src/features.R"); source("src/em_gmm.R")
L <- load_users(); pr <- readRDS("reports/results/profils.rds")

# nu = part d'atypiques visée (5 %). gamma choisi par validation croisée 5 plis sur le train : le taux
# d'atypiques sur les plis mis de côté doit rester proche de nu (un gamma trop grand surapprend la frontière)
NU <- 0.05
grid <- expand.grid(nu = c(0.02, 0.05, 0.1), gamma = c(0.01, 0.02, 0.05, 0.11, 0.25))
set.seed(SEED); fold <- sample(rep(1:5, length.out = nrow(L$ztr)))
oc <- function(z, nu, g) svm(z, type = "one-classification", kernel = "radial", nu = nu, gamma = g)
for (i in seq_len(nrow(grid))) {
  grid$atyp_cv[i] <- mean(unlist(lapply(1:5, function(f)
    !predict(oc(L$ztr[fold != f, ], grid$nu[i], grid$gamma[i]), L$ztr[fold == f, ]))))
  m <- oc(L$ztr, grid$nu[i], grid$gamma[i])
  grid$atyp_train[i] <- mean(!predict(m, L$ztr)); grid$atyp_test[i] <- mean(!predict(m, L$zte))
}
print(grid, digits = 3); write.csv(grid, "reports/results/ocsvm_grille.csv", row.names = FALSE)
g_nu <- grid[grid$nu == NU, ]
GAMMA <- g_nu$gamma[which.min(abs(g_nu$atyp_cv - NU))]
cat("gamma retenu (CV) :", GAMMA, "\n")
m <- oc(L$ztr, NU, GAMMA)
u <- rbind(cbind(L$tr, profil = pr$tr$profil), cbind(L$te, profil = pr$te$profil))
u$atypique <- !c(predict(m, L$ztr), predict(m, L$zte))
# Comparaison avec la densité du GMM : 5 % des usagers les moins vraisemblables
ll <- predict_gmm(pr$fit, rbind(L$ztr, L$zte))$ll_i
u$faible_densite_gmm <- ll <= quantile(ll[u$split == "train"], 0.05)
cat("Atypiques train/test :", round(mean(u$atypique[u$split == "train"]), 3), round(mean(u$atypique[u$split == "test"]), 3), "\n")
print(table(ocsvm = u$atypique, gmm_faible_densite = u$faible_densite_gmm))
cat("Atypiques par profil (part) :\n"); print(round(tapply(u$atypique, u$profil, mean), 3))
comp <- u %>% group_by(atypique) %>% summarise(usagers = n(), sessions_med = median(n),
  across(all_of(CLUST_VARS), ~ round(median(.x), 2)), .groups = "drop")
print(as.data.frame(comp)); write.csv(comp, "reports/results/ocsvm_atypiques.csv", row.names = FALSE)

p <- readRDS("reports/results/pca.rds")
sc <- data.frame(predict(p, rbind(L$ztr, L$zte))[, 1:2], atypique = u$atypique, split = u$split)
g <- ggplot(sc, aes(PC1, PC2, colour = atypique, shape = split)) + geom_point(alpha = .7) +
  scale_colour_manual(values = c(`FALSE` = "grey60", `TRUE` = "red")) +
  labs(title = paste0("One-class SVM (nu = ", NU, ", gamma = ", GAMMA, ") : usagers atypiques dans le plan PCA")) + theme_minimal()
ggsave("reports/figures/ocsvm.png", g, width = 7, height = 5, dpi = 110)
saveRDS(list(nu = NU, gamma = GAMMA, grid = grid, atyp = u[, c("userID", "split", "profil", "atypique", "faible_densite_gmm")]),
        "reports/results/ocsvm.rds")
