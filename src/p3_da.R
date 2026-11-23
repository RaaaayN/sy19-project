# Projet 3, étape 5 : analyse discriminante linéaire et quadratique. Rscript src/p3_da.R
suppressMessages(library(MASS))
source("src/p3_eval.R")
x <- build_p3(); m <- p3_matrix(x)
# LDA : covariance commune aux deux classes -> frontière linéaire (proche de la logistique).
lda_fit <- lda(m$X[m$tr, ], grouping = m$y[m$tr])
print(round(lda_fit$scaling, 3))
p3_save("lda", x, predict(lda_fit, m$X)$posterior[, "1"])
# QDA : une covariance par classe -> frontière quadratique ; variables non gaussiennes (binaires,
# taux bornés) donc hypothèse fragile, à vérifier sur la calibration.
qda_fit <- qda(m$X[m$tr, ], grouping = m$y[m$tr])
p3_save("qda", x, predict(qda_fit, m$X)$posterior[, "1"])
