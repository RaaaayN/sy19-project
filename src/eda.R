# Analyse exploratoire : Rscript src/eda.R (figures dans reports/figures/, chiffres dans reports/results/eda_*.csv)
source("src/features.R")
suppressMessages({ library(ggplot2); library(patchwork); library(tidyr) })
theme_set(theme_minimal(base_size = 11))
s <- load_sessions()
fig <- function(p, f, w = 9, h = 4.5) ggsave(file.path("reports/figures", f), p, width = w, height = h, dpi = 120)

# Résumé par site et par découpage
eda <- s |> group_by(site, split) |>
  summarise(n = n(), dur_moy = mean(duration), dur_med = median(duration), kwh_moy = mean(energy),
            kwh_med = median(energy), pct_connu = mean(known), pct_decl = mean(has_decl),
            pct_hist = mean(has_hist), .groups = "drop")
save_results(eda, "eda_resume.csv")

# Corrélations des cibles avec les variables numériques (train)
tr <- s[s$split == "train", ]
num <- c("hour", "dow", "weekend", "known", "has_hist", HIST_VARS, DECL_VARS)
cors <- data.frame(variable = num, cor_duree = sapply(num, \(v) cor(tr[[v]], tr$duration)),
                   cor_energie = sapply(num, \(v) cor(tr[[v]], tr$energy)))
save_results(cors, "eda_correlations.csv")

# 1. Distributions des cibles par site
long <- pivot_longer(s, c(duration, energy), names_to = "cible")
fig(ggplot(long, aes(value, fill = site)) + geom_histogram(bins = 60, alpha = .6, position = "identity") +
      facet_wrap(~cible, scales = "free", labeller = as_labeller(c(duration = "Durée (h)", energy = "Énergie (kWh)"))) +
      labs(x = NULL, y = "Sessions"), "eda_distributions.png")

# 2. Durée et énergie selon l'heure d'arrivée et le jour
p1 <- ggplot(s, aes(floor(hour), duration)) + geom_boxplot(aes(group = floor(hour)), outlier.size = .2) +
  labs(x = "Heure d'arrivée (locale)", y = "Durée (h)")
p2 <- ggplot(s, aes(factor(dow), energy)) + geom_boxplot(outlier.size = .2) +
  labs(x = "Jour (1 = lundi)", y = "Énergie (kWh)")
fig(p1 + p2, "eda_heure_jour.png")

# 3. Arrivées par heure et par site
fig(ggplot(s, aes(hour, fill = site)) + geom_histogram(binwidth = .5, alpha = .7) +
      facet_wrap(~site, scales = "free_y") + labs(x = "Heure d'arrivée", y = "Sessions"), "eda_arrivees.png", h = 3.5)

# 4. Déclarations et historique contre la réalité (usagers concernés)
dec <- s[s$has_decl == 1, ]
p3 <- ggplot(dec, aes(decl_dur, duration)) + geom_bin2d(bins = 60) + geom_abline(colour = "red") +
  labs(x = "Durée déclarée (h)", y = "Durée réelle (h)")
p4 <- ggplot(dec, aes(decl_kwh, energy)) + geom_bin2d(bins = 60) + geom_abline(colour = "red") +
  labs(x = "Énergie déclarée (kWh)", y = "Énergie livrée (kWh)")
hi <- s[s$has_hist == 1, ]
p5 <- ggplot(hi, aes(h_dur_mean, duration)) + geom_bin2d(bins = 60) + geom_abline(colour = "red") +
  labs(x = "Durée moyenne passée (h)", y = "Durée réelle (h)")
p6 <- ggplot(hi, aes(h_kwh_mean, energy)) + geom_bin2d(bins = 60) + geom_abline(colour = "red") +
  labs(x = "Énergie moyenne passée (kWh)", y = "Énergie livrée (kWh)")
fig((p3 + p4) / (p5 + p6) & scale_fill_viridis_c(trans = "log10"), "eda_declarations_historique.png", h = 8)

# 5. Volume mensuel (justifie le découpage temporel)
fig(ggplot(s, aes(floor_date(conn, "month"), fill = split)) + geom_bar() +
      labs(x = NULL, y = "Sessions par mois", fill = "Découpage"), "eda_decoupage.png", h = 3.5)
