# Tableau comparatif de tous les modèles : Rscript src/compare.R (après les scripts models_*.R)
suppressMessages({ library(dplyr); library(ggplot2) })
files <- c("baselines", "lineaires", "pca", "gam", "arbres", "svr", "mlp")
r <- bind_rows(lapply(files, \(f) read.csv(file.path("reports/results", paste0(f, ".csv")))[, c("model", "target", "split", "MAE", "RMSE", "R2")]))
tab <- r |> filter(split == "test") |> select(-split) |> arrange(target, RMSE)
write.csv(tab, "reports/results/comparatif.csv", row.names = FALSE)
print(tab, digits = 3)
ord <- tab |> filter(target == "duration") |> pull(model)
p <- ggplot(tab, aes(RMSE, factor(model, levels = rev(ord)))) + geom_col(fill = "steelblue") +
  geom_point(aes(x = MAE), shape = 4, size = 2.5) + facet_wrap(~target, scales = "free_x",
  labeller = as_labeller(c(duration = "Durée (h)", energy = "Énergie (kWh)"))) +
  labs(x = "RMSE sur le test (barres), MAE (croix)", y = NULL) + theme_minimal()
ggsave("reports/figures/comparatif.png", p, width = 9, height = 4.5, dpi = 120)
