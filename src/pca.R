# Étape 2 : standardisation (apprise sur le train) et PCA. Rscript src/pca.R
suppressPackageStartupMessages({ library(ggplot2); library(patchwork) })
source("src/features.R")
L <- load_users()
p <- prcomp(L$ztr, center = FALSE, scale. = FALSE)   # déjà centrée-réduite
ve <- data.frame(pc = seq_along(p$sdev), var = p$sdev^2 / sum(p$sdev^2))
ve$cumul <- cumsum(ve$var)
print(round(ve, 3)); write.csv(round(ve, 4), "reports/results/pca_variance.csv", row.names = FALSE)
load <- round(p$rotation[, 1:4], 3); print(load)
write.csv(load, "reports/results/pca_loadings.csv")

sc <- rbind(data.frame(predict(p, L$ztr)[, 1:2], split = "train"),
            data.frame(predict(p, L$zte)[, 1:2], split = "test"))
ar <- data.frame(var = rownames(p$rotation), p$rotation[, 1:2] * 3)
g1 <- ggplot(ve, aes(pc, var)) + geom_col(fill = "steelblue") + geom_line(aes(y = cumul)) + geom_point(aes(y = cumul)) +
  labs(x = "Composante", y = "Part de variance (barres) et cumul", title = "Éboulis des valeurs propres") + theme_minimal()
g2 <- ggplot(sc, aes(PC1, PC2)) + geom_point(aes(colour = split), alpha = .6) +
  geom_segment(data = ar, aes(0, 0, xend = PC1, yend = PC2), arrow = arrow(length = unit(.15, "cm"))) +
  geom_text(data = ar, aes(PC1 * 1.12, PC2 * 1.12, label = var), size = 3) +
  labs(title = "Usagers dans le plan PC1-PC2 (test projeté)") + theme_minimal()
ggsave("reports/figures/pca.png", g1 + g2 + plot_layout(widths = c(1, 2)), width = 12, height = 5, dpi = 110)
saveRDS(p, "reports/results/pca.rds")
