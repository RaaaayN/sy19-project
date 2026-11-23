# Projet 3, étape 3 : baselines. Rscript src/p3_baselines.R
source("src/p3_eval.R")
x <- build_p3()
p0 <- mean(x$early[x$split == "train"])
# 1) Classe majoritaire : prédit toujours "pas de départ anticipé" (proba constante = taux du train).
p3_save("majoritaire", x, rep(p0, nrow(x)), thr = 0.5)
# 2) Taux historique de l'usager : proportion lissée de ses départs anticipés passés (sessions terminées).
p3_save("taux_usager", x, x$past_rate)
