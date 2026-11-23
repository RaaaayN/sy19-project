# Projet 3 : protocole commun (matrice de design, métriques, sauvegarde des prédictions).
# Protocole : hyperparamètres par CV sur le train seulement ; seuil de décision choisi sur la
# validation (max F1) ; métriques finales sur le test (déc. 2019 - fév. 2020), jamais utilisé pour choisir.
suppressMessages({ library(pROC) })
source("src/p3_data.R")

P3_FORMULA <- early ~ site + arr_hour + dep_hour + dur_decl + weekend + kWhRequested + WhPerMile +
  log1p(n_past) + past_rate + past_gap

# Matrice numérique standardisée avec les moyennes/écarts-types du train (pas de fuite).
p3_matrix <- function(x, formula = P3_FORMULA) {
  X <- model.matrix(formula, x)[, -1, drop = FALSE]
  tr <- x$split == "train"
  mu <- colMeans(X[tr, ]); s <- apply(X[tr, ], 2, sd)
  X <- scale(X, mu, s)
  list(X = X, y = x$early, tr = tr, val = x$split == "val", te = x$split == "test")
}

p3_auc <- function(y, p) as.numeric(auc(roc(y, p, quiet = TRUE, direction = "<")))

p3_metrics <- function(y, p, thr) {
  yhat <- as.integer(p >= thr)
  tp <- sum(yhat & y); fp <- sum(yhat & !y); fn <- sum(!yhat & y); tn <- sum(!yhat & !y)
  prec <- if (tp + fp > 0) tp / (tp + fp) else 0; rec <- tp / (tp + fn)
  pc <- pmin(pmax(p, 1e-6), 1 - 1e-6)
  data.frame(auc = p3_auc(y, p), f1 = if (prec + rec > 0) 2 * prec * rec / (prec + rec) else 0,
             precision = prec, recall = rec, accuracy = (tp + tn) / length(y),
             brier = mean((p - y)^2), logloss = -mean(y * log(pc) + (1 - y) * log(1 - pc)),
             seuil = thr, tp = tp, fp = fp, fn = fn, tn = tn)
}

# Seuil qui maximise le F1 sur la validation.
p3_best_thr <- function(y, p) {
  g <- unique(quantile(p, seq(0.01, 0.99, 0.01), names = FALSE))
  f1 <- sapply(g, function(t) p3_metrics(y, p, t)$f1)
  g[which.max(f1)]
}

# p : probabilités prédites pour toutes les lignes de x (train compris ou NA).
p3_save <- function(name, x, p, extra = list(), thr = NULL) {
  dir.create("reports/preds", recursive = TRUE, showWarnings = FALSE)
  dir.create("reports/metrics/p3_models", recursive = TRUE, showWarnings = FALSE)
  keep <- x$split != "train"
  pr <- data.frame(sessionID = x$sessionID[keep], split = x$split[keep], y = x$early[keep], p = p[keep])
  saveRDS(pr, sprintf("reports/preds/p3_%s.rds", name))
  v <- pr$split == "val"
  if (is.null(thr)) thr <- p3_best_thr(pr$y[v], pr$p[v])
  m <- rbind(cbind(modele = name, jeu = "val", p3_metrics(pr$y[v], pr$p[v], thr)),
             cbind(modele = name, jeu = "test", p3_metrics(pr$y[!v], pr$p[!v], thr)))
  for (k in names(extra)) m[[k]] <- extra[[k]]
  write.csv(m, sprintf("reports/metrics/p3_models/%s.csv", name), row.names = FALSE)
  print(m[, c("modele", "jeu", "auc", "f1", "precision", "recall", "brier", "seuil")], digits = 3)
  invisible(m)
}
