# Export pour le simulateur IA04 (Go) : Rscript src/export.R (après src/models_mlp.R)
#  - reports/model_mlp.json : architecture, poids et normalisations du MLP retenu (JSON lisible par encoding/json)
#  - reports/predictions.csv : prédictions durée/énergie des sessions de validation et de test
source("src/predict.R")
suppressMessages(library(torch))
s <- load_sessions()
X <- model.matrix(model_formula("duration"), s)[, -1]
tr <- s$split == "train"
net <- torch_load("models/mlp.pt")
sd_ <- net$state_dict()
lin <- grep("weight$", names(sd_), value = TRUE)
mat <- function(t) as.matrix(as_array(t))
m <- list(
  description = "MLP durée (h) / énergie (kWh) à l'arrivée. y = denorm(L3(relu(L2(relu(L1(norm(x))))))), Lk(h) = W h + b, W de taille [sorties x entrées]",
  features = colnames(X), targets = TARGETS,
  x_mean = unname(colMeans(X[tr, ])), x_sd = unname(apply(X[tr, ], 2, sd)),
  y_mean = unname(colMeans(s[tr, TARGETS])), y_sd = unname(sapply(s[tr, TARGETS], sd)),
  sites = ACN_SITES, timezone = TZ_LOCAL,
  layers = list(W = lapply(lin, \(n) mat(sd_[[n]])),
                b = lapply(sub("weight$", "bias", lin), \(n) as.numeric(as_array(sd_[[n]]))),
                activation = c(rep("relu", length(lin) - 1), "identity")))
write_json(m, MODEL_JSON, digits = NA, auto_unbox = TRUE, pretty = FALSE)

mj <- load_model()
p <- mlp_forward(mj, X)
p_torch <- with_no_grad(as.matrix(net(torch_tensor(scale(X, mj$x_mean, mj$x_sd), dtype = torch_float()))))
p_torch <- sweep(sweep(p_torch, 2, mj$y_sd, "*"), 2, mj$y_mean, "+")
stopifnot(max(abs(p - p_torch)) < 1e-3)   # le JSON reproduit le réseau torch
k <- s$split != "train"
out <- data.frame(sessionID = s$sessionID[k], site = s$site[k], split = s$split[k],
                  connectionTime = format(s$conn[k], "%Y-%m-%dT%H:%M:%S%z"),
                  duration_true_h = round(s$duration[k], 4), duration_pred_h = round(p[k, "duration"], 4),
                  energy_true_kwh = round(s$energy[k], 4), energy_pred_kwh = round(p[k, "energy"], 4),
                  declared_kwh = ifelse(s$has_decl[k] == 1, s$decl_kwh[k], NA),
                  declared_duration_h = ifelse(s$has_decl[k] == 1, round(s$decl_dur[k], 4), NA))
write.csv(out, "reports/predictions.csv", row.names = FALSE)
print(rbind(eval_rows("MLP (JSON)", "duration", s, p[, 1], "test"), eval_rows("MLP (JSON)", "energy", s, p[, 2], "test")))
