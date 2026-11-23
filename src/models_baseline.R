# Baselines : Rscript src/models_baseline.R -> reports/results/baselines.csv
source("src/features.R")
s <- load_sessions()
tr <- s$split == "train"
hist_col <- c(duration = "h_dur_mean", energy = "h_kwh_mean")
decl_col <- c(duration = "decl_dur", energy = "decl_kwh")

res <- do.call(rbind, lapply(TARGETS, function(y) {
  mu <- mean(s[[y]][tr])                                   # moyenne globale du train
  p_user <- ifelse(s$has_hist == 1, s[[hist_col[y]]], mu)  # moyenne des sessions passées de l'usager
  p_decl <- ifelse(s$has_decl == 1, s[[decl_col[y]]], p_user)  # déclaration brute, sinon historique
  rbind(eval_rows("Moyenne globale", y, s, rep(mu, nrow(s))),
        eval_rows("Moyenne par usager", y, s, p_user),
        eval_rows("Déclaration usager", y, s, p_decl))
}))
save_results(res, "baselines.csv")
