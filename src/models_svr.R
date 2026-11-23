# SVR à noyau RBF (e1071) : Rscript src/models_svr.R
# Le coût est en O(n^2) à O(n^3) : on sous-échantillonne N_FIT sessions du train pour l'ajustement final
# et N_TUNE pour la grille (C, gamma), réglée par CV 5 plis. Variables centrées-réduites par svm().
source("src/features.R")
suppressMessages(library(e1071))
set.seed(19)
N_TUNE <- 4000; N_FIT <- 10000
s <- load_sessions()
itr <- which(s$split == "train")
sub_fit <- sample(itr, N_FIT); sub_tune <- sample(sub_fit, N_TUNE)
res <- list(); grids <- list()
for (y in TARGETS) {
  f <- model_formula(y)
  t0 <- Sys.time()
  tu <- tune(svm, f, data = s[sub_tune, c(y, ALL_VARS)], kernel = "radial",  # colonnes utiles seulement (NA ailleurs)
             ranges = list(cost = c(0.3, 1, 3, 10), gamma = c(0.01, 0.03, 0.1, 0.3)),
             tunecontrol = tune.control(sampling = "cross", cross = 5))
  grids[[y]] <- cbind(target = y, tu$performances)
  bp <- tu$best.parameters
  fit <- svm(f, data = s[sub_fit, c(y, ALL_VARS)], kernel = "radial", cost = bp$cost, gamma = bp$gamma)
  res[[y]] <- cbind(eval_rows("SVR (RBF)", y, s, predict(fit, s[, ALL_VARS])),  # s[, ALL_VARS] : predict.svm écarte les lignes ayant un NA ailleurs
                    param = sprintf("C=%g, gamma=%g, n=%d, %d SV", bp$cost, bp$gamma, N_FIT, fit$tot.nSV),
                    secondes = round(as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
save_results(do.call(rbind, res), "svr.csv")
g <- do.call(rbind, grids); write.csv(g, "reports/results/svr_grille.csv", row.names = FALSE); print(g)
