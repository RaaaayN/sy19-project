# Projet 3, étape 4 : régression logistique et logistique L1 (lasso). Rscript src/p3_logit.R
suppressMessages(library(glmnet))
source("src/p3_eval.R")
set.seed(19)
x <- build_p3(); m <- p3_matrix(x)
dtr <- data.frame(m$X[m$tr, ], early = m$y[m$tr])

# 1) Logistique sans pénalité, variables standardisées (coefficients comparables).
fit <- glm(early ~ ., family = binomial, data = dtr)
print(summary(fit)$coefficients, digits = 3)
write.csv(data.frame(variable = rownames(summary(fit)$coefficients), summary(fit)$coefficients, check.names = FALSE),
          "reports/metrics/p3_logit_coefs.csv", row.names = FALSE)
p3_save("logistique", x, predict(fit, data.frame(m$X), type = "response"))

# 2) Logistique L1 : lambda par CV 10 plis sur le train (critère : déviance binomiale), règle 1-SE.
cv <- cv.glmnet(m$X[m$tr, ], m$y[m$tr], family = "binomial", alpha = 1, nfolds = 10)
cf <- as.matrix(coef(cv, s = "lambda.1se"))
cat("lambda.min =", cv$lambda.min, " lambda.1se =", cv$lambda.1se, "\n"); print(cf, digits = 3)
write.csv(data.frame(variable = rownames(cf), coef_lambda_1se = cf[, 1],
                     coef_lambda_min = as.matrix(coef(cv, s = "lambda.min"))[, 1]),
          "reports/metrics/p3_lasso_coefs.csv", row.names = FALSE)
png("reports/figures/p3_lasso_cv.png", width = 700, height = 450); plot(cv); dev.off()
png("reports/figures/p3_lasso_chemin.png", width = 700, height = 450)
plot(cv$glmnet.fit, xvar = "lambda", label = TRUE); abline(v = log(cv$lambda.1se), lty = 2); dev.off()
p3_save("lasso", x, as.numeric(predict(cv, m$X, s = "lambda.1se", type = "response")),
        extra = list(lambda = cv$lambda.1se, n_var = sum(cf[-1, 1] != 0)))
