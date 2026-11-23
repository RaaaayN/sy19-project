# Perceptron multicouche (torch pour R) prédisant conjointement durée et énergie : Rscript src/models_mlp.R
# Entrées et cibles centrées-réduites (moyennes et écarts-types du train).
# Grille (largeur, dropout) choisie par CV 3 plis sur le train, early stopping sur le pli de validation ;
# modèle final entraîné sur le train avec early stopping sur la période de validation (sept.-nov. 2019).
source("src/features.R")
suppressMessages({ library(torch); library(ggplot2) })
set.seed(19); torch_manual_seed(19)
s <- load_sessions()
X <- model.matrix(model_formula("duration"), s)[, -1]
tr <- which(s$split == "train"); va <- which(s$split == "val")
mx <- colMeans(X[tr, ]); sx <- apply(X[tr, ], 2, sd)
Xs <- scale(X, mx, sx)
Y <- as.matrix(s[, TARGETS]); my <- colMeans(Y[tr, ]); sy <- apply(Y[tr, ], 2, sd)
Ys <- scale(Y, my, sy)
tt <- function(m) torch_tensor(m, dtype = torch_float())

make_net <- function(width, dropout) nn_sequential(
  nn_linear(ncol(X), width), nn_relu(), nn_dropout(dropout),
  nn_linear(width, width), nn_relu(), nn_dropout(dropout),
  nn_linear(width, length(TARGETS)))

# Entraîne sur i_fit, early stopping sur i_val (patience en époques) ; renvoie le réseau et la courbe
train_net <- function(i_fit, i_val, width, dropout, max_epochs = 150, patience = 10, bs = 512) {
  net <- make_net(width, dropout); opt <- optim_adam(net$parameters, lr = 1e-3, weight_decay = 1e-5)
  xf <- tt(Xs[i_fit, ]); yf <- tt(Ys[i_fit, ]); xv <- tt(Xs[i_val, ]); yv <- tt(Ys[i_val, ])
  best <- Inf; best_state <- NULL; wait <- 0; curve <- NULL
  for (ep in seq_len(max_epochs)) {
    net$train(); perm <- sample(length(i_fit)); tl <- 0
    for (b in split(perm, ceiling(seq_along(perm) / bs))) {
      opt$zero_grad(); loss <- nnf_mse_loss(net(xf[b, ]), yf[b, ]); loss$backward(); opt$step()
      tl <- tl + loss$item() * length(b)
    }
    net$eval(); vl <- with_no_grad(nnf_mse_loss(net(xv), yv)$item())
    curve <- rbind(curve, data.frame(epoch = ep, train = tl / length(i_fit), val = vl))
    if (vl < best - 1e-4) { best <- vl; wait <- 0; best_state <- lapply(net$state_dict(), \(p) p$clone()) }
    else if ((wait <- wait + 1) >= patience) break
  }
  net$load_state_dict(best_state); net$eval()
  list(net = net, best = best, curve = curve)
}

grid <- expand.grid(width = c(32, 128), dropout = c(0, 0.2))
folds <- sample(rep(1:3, length.out = length(tr)))
grid$cv_mse <- sapply(seq_len(nrow(grid)), function(g) mean(sapply(1:3, function(k)
  train_net(tr[folds != k], tr[folds == k], grid$width[g], grid$dropout[g])$best)))
print(grid); write.csv(grid, "reports/results/mlp_grille.csv", row.names = FALSE)

bg <- grid[which.min(grid$cv_mse), ]
fit <- train_net(tr, va, bg$width, bg$dropout)
pred <- with_no_grad(as.matrix(fit$net(tt(Xs))))
pred <- sweep(sweep(pred, 2, sy, "*"), 2, my, "+")
res <- do.call(rbind, lapply(seq_along(TARGETS), function(j)
  cbind(eval_rows("MLP (torch)", TARGETS[j], s, pred[, j]),
        param = sprintf("2x%d ReLU, dropout %.1f, %d époques", bg$width, bg$dropout, which.min(fit$curve$val)))))
save_results(res, "mlp.csv")
dir.create("models", showWarnings = FALSE)
torch_save(fit$net, "models/mlp.pt")
write.csv(fit$curve, "reports/results/mlp_courbe.csv", row.names = FALSE)
ggsave("reports/figures/mlp_early_stopping.png",
       ggplot(tidyr::pivot_longer(fit$curve, c(train, val)), aes(epoch, value, colour = name)) + geom_line() +
         geom_vline(xintercept = which.min(fit$curve$val), linetype = 2) +
         labs(x = "Époque", y = "MSE (cibles centrées-réduites)", colour = NULL) + theme_minimal(),
       width = 7, height = 4, dpi = 120)
