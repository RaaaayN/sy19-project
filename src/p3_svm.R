# Projet 3, étape 8 : SVM linéaire et RBF (e1071 / libsvm). Rscript src/p3_svm.R
# Coût : libsvm est en O(n^2)-O(n^3) ; on sous-échantillonne le train (documenté dans le rapport) :
# 8 000 sessions pour la grille en CV 3 plis ; l'ajustement final utilise tout le train.
suppressMessages(library(e1071))
source("src/p3_eval.R")
set.seed(19)
x <- build_p3(); m <- p3_matrix(x)
itr <- which(m$tr)
i_cv <- sample(itr, 8000); i_fit <- itr
folds <- sample(rep(1:3, length.out = length(i_cv)))

# AUC moyenne en CV (scores = valeurs de décision, donc sans calibration nécessaire).
cv_auc <- function(...) {
  mean(sapply(1:3, function(k) {
    a <- i_cv[folds != k]; b <- i_cv[folds == k]
    f <- svm(m$X[a, ], factor(m$y[a]), ...)
    dv <- attr(predict(f, m$X[b, ], decision.values = TRUE), "decision.values")
    if (colnames(dv) == "0/1") dv <- -dv       # colonne "0/1" : valeur > 0 favorise la classe 0
    p3_auc(m$y[b], dv)
  }))
}
g_lin <- data.frame(cost = c(0.01, 0.1, 1, 10))
g_lin$auc_cv <- sapply(g_lin$cost, function(C) cv_auc(kernel = "linear", cost = C))
print(g_lin)
g_rbf <- expand.grid(cost = c(0.3, 1, 3, 10, 30), gamma = c(0.003, 0.01, 0.03, 0.1, 0.3))
g_rbf$auc_cv <- mapply(function(C, g) cv_auc(kernel = "radial", cost = C, gamma = g), g_rbf$cost, g_rbf$gamma)
print(g_rbf)
write.csv(rbind(cbind(noyau = "lineaire", gamma = NA, g_lin), cbind(noyau = "rbf", g_rbf[, c("gamma", "cost", "auc_cv")])),
          "reports/metrics/p3_svm_grille.csv", row.names = FALSE)

# Ajustement final avec probabilités (Platt, CV interne de libsvm) sur tout le train.
fit_prob <- function(...) {
  f <- svm(m$X[i_fit, ], factor(m$y[i_fit]), probability = TRUE, ...)
  attr(predict(f, m$X, probability = TRUE), "probabilities")[, "1"]
}
bl <- g_lin[which.max(g_lin$auc_cv), ]; br <- g_rbf[which.max(g_rbf$auc_cv), ]
p3_save("svm_lineaire", x, fit_prob(kernel = "linear", cost = bl$cost), extra = list(cost = bl$cost))
p3_save("svm_rbf", x, fit_prob(kernel = "radial", cost = br$cost, gamma = br$gamma),
        extra = list(cost = br$cost, gamma = br$gamma))
