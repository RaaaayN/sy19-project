# Régression linéaire, ridge et lasso : Rscript src/models_linear.R
# Hyperparamètre lambda choisi par validation croisée 10 plis sur le train uniquement.
source("src/features.R")
suppressMessages(library(glmnet))
set.seed(19)
s <- load_sessions()
X <- model.matrix(model_formula("duration"), s)[, -1]   # mêmes variables pour les deux cibles
tr <- s$split == "train"

res <- list(); coefs <- list()
png("reports/figures/lin_chemins.png", width = 1100, height = 800, res = 110)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
for (y in TARGETS) {
  fit_lm <- lm(model_formula(y), data = s[tr, ])
  res[[length(res) + 1]] <- eval_rows("Régression linéaire", y, s, predict(fit_lm, s))
  for (a in c(0, 1)) {
    nm <- if (a == 0) "Ridge" else "Lasso"
    cv <- cv.glmnet(X[tr, ], s[[y]][tr], alpha = a, nfolds = 10, lambda = 10^seq(2, -4, length.out = 100))
    res[[length(res) + 1]] <- cbind(eval_rows(nm, y, s, as.numeric(predict(cv, X, s = "lambda.min"))),
                                    lambda = cv$lambda.min)
    plot(cv$glmnet.fit, xvar = "lambda", main = sprintf("%s - %s", nm, y))
    abline(v = log(c(cv$lambda.min, cv$lambda.1se)), lty = 2)
    coefs[[paste(nm, y)]] <- data.frame(model = nm, target = y, variable = rownames(coef(cv)),
      coef_min = as.numeric(coef(cv, s = "lambda.min")), coef_1se = as.numeric(coef(cv, s = "lambda.1se")))
  }
  coefs[[paste("lm", y)]] <- data.frame(model = "Régression linéaire", target = y, variable = names(coef(fit_lm)),
                                        coef_min = coef(fit_lm), coef_1se = NA)
}
dev.off()
res <- do.call(rbind, lapply(res, \(r) { if (is.null(r$lambda)) r$lambda <- NA; r }))
save_results(res, "lineaires.csv")
cf <- do.call(rbind, coefs); rownames(cf) <- NULL
write.csv(cf, "reports/results/lineaires_coefs.csv", row.names = FALSE)
cat("\nVariables retenues par le lasso (lambda.1se) :\n")
for (y in TARGETS) { k <- cf$model == "Lasso" & cf$target == y & cf$coef_1se != 0 & cf$variable != "(Intercept)"
  cat(y, ":", cf$variable[k], "\n") }
