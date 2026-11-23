# PCA sur les variables d'historique et régression sur composantes : Rscript src/models_pca.R
# PCA ajustée sur les sessions du train ayant un historique ; nombre de composantes k choisi par CV 10 plis (train).
source("src/features.R")
suppressMessages({ library(ggplot2); library(patchwork) })
set.seed(19)
s <- load_sessions()
tr <- s$split == "train"
pca <- prcomp(s[tr & s$has_hist == 1, HIST_VARS], center = TRUE, scale. = TRUE)
pcs <- predict(pca, s[, HIST_VARS]) * s$has_hist          # composantes nulles sans historique
colnames(pcs) <- paste0("PC", seq_len(ncol(pcs)))
s <- cbind(s, pcs)
other <- c(BASE_VARS, "has_hist", DECL_VARS)

ve <- data.frame(k = seq_along(pca$sdev), var_expl = pca$sdev^2 / sum(pca$sdev^2))
ve$cumul <- cumsum(ve$var_expl)
write.csv(ve, "reports/results/pca_variance.csv", row.names = FALSE)
load <- data.frame(variable = HIST_VARS, round(pca$rotation[, 1:3], 3))
write.csv(load, "reports/results/pca_loadings.csv", row.names = FALSE)
print(ve, digits = 3); print(load)

folds <- sample(rep(1:10, length.out = sum(tr)))
cv_k <- expand.grid(target = TARGETS, k = 0:length(HIST_VARS))
cv_k$rmse_cv <- mapply(function(y, k) {
  f <- model_formula(y, c(other, colnames(pcs)[seq_len(k)]))
  d <- s[tr, ]
  sqrt(mean(unlist(lapply(1:10, function(i) (d[[y]][folds == i] - predict(lm(f, d[folds != i, ]), d[folds == i, ]))^2))))
}, as.character(cv_k$target), cv_k$k)
write.csv(cv_k, "reports/results/pca_cv.csv", row.names = FALSE)

res <- do.call(rbind, lapply(TARGETS, function(y) {
  cy <- cv_k[cv_k$target == y, ]; k <- cy$k[which.min(cy$rmse_cv)]
  fit <- lm(model_formula(y, c(other, colnames(pcs)[seq_len(k)])), s[tr, ])
  cbind(eval_rows("Régression sur composantes (PCR)", y, s, predict(fit, s)), k = k)
}))
save_results(res, "pca.csv")

p1 <- ggplot(ve, aes(k, var_expl)) + geom_col() + geom_line(aes(y = cumul)) + geom_point(aes(y = cumul)) +
  labs(x = "Composante", y = "Variance expliquée (barres) et cumulée")
p2 <- ggplot(load, aes(PC1, PC2, label = variable)) + geom_segment(aes(xend = 0, yend = 0), colour = "grey50") +
  geom_text(size = 3) + coord_equal() + labs(title = "Poids des variables (PC1, PC2)")
p3 <- ggplot(cv_k, aes(k, rmse_cv, colour = target)) + geom_line() + geom_point() +
  facet_wrap(~target, scales = "free_y") + labs(x = "Nombre de composantes", y = "RMSE (CV 10 plis)") +
  theme(legend.position = "none")
ggsave("reports/figures/pca.png", (p1 + p2) / p3, width = 10, height = 8, dpi = 120)
