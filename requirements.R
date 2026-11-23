# Installe les paquets R du projet : Rscript requirements.R
options(repos = "https://cloud.r-project.org")
pk <- c("httr", "jsonlite", "dplyr", "tidyr", "ggplot2", "lubridate", "patchwork",
        "glmnet", "mgcv", "MASS", "rpart", "ipred", "randomForest", "e1071", "kernlab",
        "mclust", "cluster", "nnet", "pROC", "yardstick", "testthat", "rmarkdown", "knitr", "torch")
install.packages(setdiff(pk, rownames(installed.packages())))
# torch : après installation, lancer une fois torch::install_torch() (télécharge libtorch)
