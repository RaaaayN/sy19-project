# Arbre CART, bagging et random forest : Rscript src/models_trees.R
# Arbre : cp choisi par CV 10 plis interne à rpart (règle 1-SE) sur le train.
# Random forest : mtry choisi par erreur out-of-bag sur le train. Bagging = 50 arbres non élagués (ipred).
source("src/features.R")
suppressMessages({ library(rpart); library(ipred); library(randomForest); library(ggplot2) })
set.seed(19)
s <- load_sessions()
tr <- s$split == "train"
dir.create("models", showWarnings = FALSE)
res <- list(); imp <- list(); oob <- list()
for (y in TARGETS) {
  f <- model_formula(y)
  full <- rpart(f, s[tr, ], control = rpart.control(cp = 1e-4, xval = 10))
  ct <- full$cptable; thr <- min(ct[, "xerror"]) + ct[which.min(ct[, "xerror"]), "xstd"]
  cp <- ct[which(ct[, "xerror"] <= thr)[1], "CP"]
  tree <- prune(full, cp = cp)
  res[[paste(y, "tree")]] <- cbind(eval_rows("Arbre CART", y, s, predict(tree, s)), param = sprintf("cp=%.5f, %d feuilles", cp, sum(tree$frame$var == "<leaf>")))
  png(sprintf("reports/figures/arbre_%s.png", y), width = 1400, height = 800, res = 110)
  plot(prune(full, cp = max(cp, ct[min(nrow(ct), 9), "CP"])), uniform = TRUE, margin = .05)  # arbre affiché limité à ~8 coupures
  text(prune(full, cp = max(cp, ct[min(nrow(ct), 9), "CP"])), cex = .7, use.n = FALSE); dev.off()

  bag <- bagging(f, s[tr, ], nbagg = 50, coob = FALSE, control = rpart.control(cp = 0, minsplit = 10, xval = 0))
  res[[paste(y, "bag")]] <- cbind(eval_rows("Bagging", y, s, predict(bag, s)), param = "50 arbres")

  grid <- c(3, 5, 8, 12)
  fits <- lapply(grid, \(m) randomForest(f, s[tr, ], ntree = 300, mtry = m, importance = TRUE))
  oob[[y]] <- data.frame(target = y, mtry = grid, rmse_oob = sapply(fits, \(r) sqrt(tail(r$mse, 1))))
  rf <- fits[[which.min(oob[[y]]$rmse_oob)]]
  saveRDS(rf, sprintf("models/rf_%s.rds", y))
  res[[paste(y, "rf")]] <- cbind(eval_rows("Random forest", y, s, predict(rf, s)), param = sprintf("mtry=%d, 300 arbres", rf$mtry))
  im <- importance(rf, type = 1)
  imp[[y]] <- data.frame(target = y, variable = rownames(im), inc_mse = im[, 1])
}
save_results(do.call(rbind, res), "arbres.csv")
o <- do.call(rbind, oob); write.csv(o, "reports/results/rf_oob.csv", row.names = FALSE); print(o)
im <- do.call(rbind, imp); rownames(im) <- NULL
write.csv(im, "reports/results/rf_importance.csv", row.names = FALSE)
ggsave("reports/figures/rf_importance.png",
       ggplot(im, aes(inc_mse, reorder(variable, inc_mse))) + geom_col() + facet_wrap(~target, scales = "free_x") +
         labs(x = "Augmentation du MSE par permutation (%IncMSE)", y = NULL) + theme_minimal(),
       width = 9, height = 5, dpi = 120)
