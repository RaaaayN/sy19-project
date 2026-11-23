# Étape 9 : investigation originale.
#  A. Sensibilité au seuil minimal de sessions par usager (K choisi par BIC, accord avec la partition de référence)
#  B. Stabilité du profil d'un même usager entre deux périodes
# Rscript src/investigation.R
suppressPackageStartupMessages({ library(mclust); library(cluster); library(ggplot2); library(patchwork); library(dplyr) })
source("src/features.R"); source("src/em_gmm.R")
s <- clean_sessions(load_acn())
L <- load_users(); pr <- readRDS("reports/results/profils.rds")
ctr <- attr(L$ztr, "center"); scl <- attr(L$ztr, "scl")
ref_assign <- function(u) predict_gmm(pr$fit, clust_matrix(u, ctr, scl))$cluster

# ---- A. seuil minimal ----
u_all <- user_features(s, 3)
commun <- u_all$userID[u_all$n >= 30]          # usagers présents quelle que soit la valeur du seuil
ref_commun <- ref_assign(u_all[u_all$userID %in% commun, ])
A <- do.call(rbind, lapply(c(3, 5, 10, 20, 30), function(m) {
  u <- u_all[u_all$n >= m, ]; z <- clust_matrix(u)
  fits <- lapply(1:5, function(K) em_gmm(z, K, n_init = 20))
  bic <- sapply(fits, function(f) if (is.null(f)) NA else f$bic)
  K <- which.min(bic); f <- fits[[K]]
  ic <- u$userID %in% commun
  set.seed(SEED); km <- kmeans(z, 4, nstart = 50)
  data.frame(seuil = m, usagers = nrow(u), K_BIC = K,
             ari_ref_commun = adjustedRandIndex(f$cluster[ic], ref_commun),
             incertitude_moy = mean(1 - apply(f$resp, 1, max)),
             sil_kmeans4 = mean(silhouette(km$cluster, dist(z))[, 3]))
}))
print(A, digits = 3); write.csv(A, "reports/results/investigation_seuil.csv", row.names = FALSE)

# ---- B. deux périodes ----
coupure <- as.Date("2019-07-01")
p1 <- user_features(s[s$day < coupure, ], MIN_SESSIONS)
p2 <- user_features(s[s$day >= coupure, ], MIN_SESSIONS)
ids <- intersect(p1$userID, p2$userID)
c1 <- ref_assign(p1[match(ids, p1$userID), ]); c2 <- ref_assign(p2[match(ids, p2$userID), ])
tr_mat <- table(periode1 = pr$noms[c1], periode2 = pr$noms[c2])
accord <- mean(c1 == c2)
hasard <- sum(prop.table(table(c1)) * prop.table(table(factor(c2, levels = names(table(c1))))))
B <- data.frame(usagers_2_periodes = length(ids), part_meme_profil = accord, accord_hasard = hasard,
                ari = adjustedRandIndex(c1, c2), kappa = (accord - hasard) / (1 - hasard))
cat("Usagers p1/p2/communs :", nrow(p1), nrow(p2), length(ids), "\n")
print(tr_mat); print(B, digits = 3)
write.csv(B, "reports/results/investigation_periodes.csv", row.names = FALSE)
write.csv(as.data.frame.matrix(tr_mat), "reports/results/investigation_transitions.csv")

# Les variables d'un usager bougent-elles plus entre périodes qu'entre usagers ? (corrélation p1-p2)
z1 <- clust_matrix(p1[match(ids, p1$userID), ], ctr, scl); z2 <- clust_matrix(p2[match(ids, p2$userID), ], ctr, scl)
cors <- round(sapply(CLUST_VARS, function(v) cor(z1[, v], z2[, v])), 2)
print(cors); write.csv(data.frame(variable = names(cors), cor_p1_p2 = cors), "reports/results/investigation_correlations.csv", row.names = FALSE)

gA <- ggplot(A, aes(seuil, ari_ref_commun)) + geom_line() + geom_point() + geom_text(aes(label = paste0("K=", K_BIC)), vjust = -1) +
  scale_y_continuous(limits = c(0, 1)) + labs(x = "seuil minimal de sessions", y = "ARI avec la référence (usagers n >= 30)",
  title = "A. Sensibilité au seuil") + theme_minimal()
gB <- ggplot(as.data.frame(tr_mat), aes(periode2, periode1, fill = Freq)) + geom_tile() + geom_text(aes(label = Freq)) +
  scale_fill_gradient(low = "white", high = "steelblue") + labs(title = "B. Profil avant / après juillet 2019") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 20, hjust = 1))
ggsave("reports/figures/investigation.png", gA + gB, width = 12, height = 4.5, dpi = 110)
