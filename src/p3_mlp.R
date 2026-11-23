# Projet 3, étape 9 : perceptron multicouche (paquet R torch). Rscript src/p3_mlp.R
# Réglage (taille cachée, weight decay) et nombre d'époques sur un hold-out interne au train
# (les 20 % les plus récents du train) ; réajustement final sur tout le train.
suppressMessages(library(torch))
source("src/p3_eval.R")
set.seed(19); torch_manual_seed(19)
x <- build_p3(); m <- p3_matrix(x)
itr <- which(m$tr)                      # x est trié par date : la fin du train sert de hold-out
cut <- floor(0.8 * length(itr)); i_a <- itr[1:cut]; i_b <- itr[(cut + 1):length(itr)]
tens <- function(i) list(X = torch_tensor(m$X[i, ], dtype = torch_float()),
                         y = torch_tensor(m$y[i], dtype = torch_float())$unsqueeze(2))

make_net <- function(h) nn_sequential(nn_linear(ncol(m$X), h), nn_relu(), nn_dropout(0.2),
                                      nn_linear(h, h), nn_relu(), nn_dropout(0.2), nn_linear(h, 1))
predict_net <- function(net, X) { net$eval(); with_no_grad(as.numeric(torch_sigmoid(net(X)))) }

train_net <- function(i_train, h, wd, epochs, i_hold = NULL, bs = 256) {
  torch_manual_seed(19)
  net <- make_net(h); opt <- optim_adam(net$parameters, lr = 1e-3, weight_decay = wd)
  a <- tens(i_train); b <- if (!is.null(i_hold)) tens(i_hold)
  hist <- numeric(0)
  for (e in seq_len(epochs)) {
    net$train()
    for (bt in split(sample(length(i_train)), ceiling(seq_along(i_train) / bs))) {
      opt$zero_grad()
      loss <- nnf_binary_cross_entropy_with_logits(net(a$X[bt, ]), a$y[bt, ])
      loss$backward(); opt$step()
    }
    if (!is.null(b)) hist[e] <- p3_auc(m$y[i_hold], predict_net(net, b$X))
  }
  list(net = net, hist = hist)
}

grid <- expand.grid(h = c(16, 64), wd = c(0, 1e-3))
res <- lapply(seq_len(nrow(grid)), function(g) {
  r <- train_net(i_a, grid$h[g], grid$wd[g], epochs = 100, i_hold = i_b)
  cat("h =", grid$h[g], " wd =", grid$wd[g], " AUC hold-out max =", round(max(r$hist), 4),
      " époque =", which.max(r$hist), "\n")
  r$hist
})
grid$auc_holdout <- sapply(res, max); grid$epoques <- sapply(res, which.max)
write.csv(grid, "reports/metrics/p3_mlp_grille.csv", row.names = FALSE)
b <- grid[which.max(grid$auc_holdout), ]
png("reports/figures/p3_mlp_courbes.png", width = 750, height = 450)
matplot(sapply(res, identity), type = "l", lty = 1, col = 1:4, xlab = "époque", ylab = "AUC hold-out (fin du train)")
legend("bottomright", legend = sprintf("h=%d, wd=%g", grid$h, grid$wd), col = 1:4, lty = 1); dev.off()

fin <- train_net(itr, b$h, b$wd, epochs = b$epoques)
p3_save("mlp", x, predict_net(fin$net, torch_tensor(m$X, dtype = torch_float())),
        extra = list(h = b$h, wd = b$wd, epoques = b$epoques))
