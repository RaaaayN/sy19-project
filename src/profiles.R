# Étape 6 : interprétation et nommage des profils (GMM retenu par BIC). Rscript src/profiles.R
suppressPackageStartupMessages({ library(ggplot2); library(patchwork); library(dplyr) })
source("src/features.R"); source("src/em_gmm.R")
L <- load_users()
gm <- readRDS("reports/results/gmm.rds"); fit <- gm$fits[[gm$K_BIC]]
km <- readRDS("reports/results/kmeans.rds"); kk <- readRDS("reports/results/kkmeans.rds")

# Composantes réordonnées par heure d'arrivée médiane, puis nommées (noms choisis après lecture du tableau)
ord <- order(tapply(L$tr$arr_med, fit$cluster, median))
fit$pi <- fit$pi[ord]; fit$mu <- fit$mu[ord, , drop = FALSE]; fit$Sigma <- fit$Sigma[ord]
NOMS <- c("Pendulaire régulier", "Pendulaire flexible", "Visiteur irrégulier")[seq_along(ord)]
L$tr$profil <- factor(NOMS[predict_gmm(fit, L$ztr)$cluster], NOMS)
L$te$profil <- factor(NOMS[predict_gmm(fit, L$zte)$cluster], NOMS)

desc <- function(u, jeu) u %>% group_by(profil) %>%
  summarise(jeu = jeu, usagers = n(), part = round(n() / nrow(u), 2), sessions_med = median(n),
            across(all_of(CLUST_VARS), ~ round(median(.x), 2)),
            part_caltech = round(mean(site == "caltech"), 2), .groups = "drop")
tab <- rbind(desc(L$tr, "train"), desc(L$te, "test"))
print(as.data.frame(tab)); write.csv(tab, "reports/results/profils_gmm.csv", row.names = FALSE)

# Plan PCA coloré par méthode
p <- readRDS("reports/results/pca.rds")
sc <- data.frame(predict(p, L$ztr)[, 1:2])
pl <- function(cl, titre) ggplot(sc, aes(PC1, PC2, colour = factor(cl))) + geom_point(alpha = .7, size = 1.2) +
  labs(title = titre, colour = NULL) + theme_minimal() + theme(legend.position = "bottom")
g <- pl(L$tr$profil, paste0("GMM (K = ", gm$K_BIC, ", BIC)")) + guides(colour = guide_legend(nrow = 2)) +
  pl(km$km$cluster, paste0("k-means (k = ", km$k, ")")) + pl(kk$cl, paste0("kernel k-means (k = ", kk$k, ")"))
ggsave("reports/figures/profils_pca.png", g, width = 13, height = 4.8, dpi = 110)

# Distributions par session des usagers du train, par profil
s <- clean_sessions(load_acn()) %>% inner_join(L$tr[, c("userID", "profil")], by = "userID")
s$dep_mod <- s$dep_h %% 24
long <- rbind(data.frame(profil = s$profil, var = "Arrivée (h)", v = s$arr_h),
              data.frame(profil = s$profil, var = "Départ (h, modulo 24)", v = s$dep_mod),
              data.frame(profil = s$profil, var = "Énergie (kWh)", v = pmin(s$kWhDelivered, 60)))
g2 <- ggplot(long, aes(v, fill = profil)) + geom_density(alpha = .4, colour = NA) + facet_wrap(~var, scales = "free") +
  labs(x = NULL, y = "densité", fill = NULL, title = "Sessions des usagers du train selon leur profil") +
  theme_minimal() + theme(legend.position = "bottom")
ggsave("reports/figures/profils_sessions.png", g2, width = 12, height = 4, dpi = 110)
saveRDS(list(fit = fit, noms = NOMS, tr = L$tr[, c("userID", "profil")], te = L$te[, c("userID", "profil")]),
        "reports/results/profils.rds")
