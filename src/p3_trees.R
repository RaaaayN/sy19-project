# Projet 3, étape 7 : arbre CART, bagging, random forest. Rscript src/p3_trees.R
suppressMessages({ library(rpart); library(randomForest) })
source("src/p3_eval.R")
set.seed(19)
x <- build_p3()
vars <- c("site", "arr_hour", "dep_hour", "dur_decl", "weekend", "kWhRequested", "WhPerMile",
          "n_past", "past_rate", "past_gap")
d <- x[, vars]; y <- factor(x$early); tr <- x$split == "train"

# 1) Arbre : on fait pousser un grand arbre puis on élague avec le cp choisi par la CV 10 plis
#    interne de rpart (règle 1-SE : plus petit arbre à moins d'un écart-type du minimum).
big <- rpart(y ~ ., data = cbind(d, y = y)[tr, ], method = "class", cp = 1e-4, xval = 10)
ct <- big$cptable; i <- which.min(ct[, "xerror"])
cp1se <- ct[which(ct[, "xerror"] <= ct[i, "xerror"] + ct[i, "xstd"])[1], "CP"]
tree <- prune(big, cp = cp1se)
cat("cp 1-SE =", cp1se, " feuilles =", sum(tree$frame$var == "<leaf>"), "\n")
png("reports/figures/p3_arbre.png", width = 1100, height = 700)
plot(tree, uniform = TRUE, margin = 0.02); text(tree, cex = 0.7, use.n = FALSE); dev.off()
p3_save("arbre", x, predict(tree, d, type = "prob")[, "1"], extra = list(cp = cp1se))

# 2) Random forest / bagging : réglage de mtry et nodesize par l'AUC out-of-bag sur le train
#    (l'OOB joue le rôle d'une validation croisée, sans toucher à la validation ni au test).
#    Bagging = random forest avec mtry = nombre de variables.
grid <- expand.grid(mtry = c(2, 3, 5, length(vars)), nodesize = c(1, 20, 50))
grid$auc_oob <- NA
for (g in seq_len(nrow(grid))) {
  set.seed(19)
  f <- randomForest(d[tr, ], y[tr], ntree = 300, mtry = grid$mtry[g], nodesize = grid$nodesize[g])
  grid$auc_oob[g] <- p3_auc(x$early[tr], f$votes[, "1"])
  print(grid[g, ])
}
write.csv(grid, "reports/metrics/p3_rf_grille.csv", row.names = FALSE)
fit_best <- function(gr) {
  b <- gr[which.max(gr$auc_oob), ]; set.seed(19)
  list(par = b, fit = randomForest(d[tr, ], y[tr], ntree = 500, mtry = b$mtry, nodesize = b$nodesize, importance = TRUE))
}
bag <- fit_best(grid[grid$mtry == length(vars), ])
rf  <- fit_best(grid[grid$mtry <  length(vars), ])
p3_save("bagging", x, predict(bag$fit, d, type = "prob")[, "1"], extra = list(mtry = bag$par$mtry, nodesize = bag$par$nodesize))
p3_save("random_forest", x, predict(rf$fit, d, type = "prob")[, "1"], extra = list(mtry = rf$par$mtry, nodesize = rf$par$nodesize))
imp <- importance(rf$fit)
write.csv(data.frame(variable = rownames(imp), imp, check.names = FALSE), "reports/metrics/p3_rf_importance.csv", row.names = FALSE)
png("reports/figures/p3_rf_importance.png", width = 900, height = 450); varImpPlot(rf$fit, main = "Random forest : importance"); dev.off()
