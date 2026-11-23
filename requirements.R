# Installe les paquets R du projet : Rscript requirements.R
options(repos = "https://cloud.r-project.org")
pk <- c("httr", "jsonlite", "dplyr", "tidyr", "ggplot2", "lubridate", "patchwork",
        "glmnet", "mgcv", "MASS", "rpart", "ipred", "randomForest", "e1071", "kernlab",
        "mclust", "cluster", "nnet", "pROC", "yardstick", "testthat", "rmarkdown", "knitr")
install.packages(setdiff(pk, rownames(installed.packages())))
# Réseau de neurones (projet 1) : torch pour R, puis torch::install_torch() une fois (télécharge libtorch)
if (!"torch" %in% rownames(installed.packages())) { install.packages("torch"); torch::install_torch() }
