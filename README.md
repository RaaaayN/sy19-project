# SY19 : sessions de recharge de véhicules électriques (ACN-Data)

Projet de l'UV SY19 (UTC, A26), en R avec un rendu Quarto/R Markdown comme l'exige le sujet.
Données : [ACN-Data](https://ev.caltech.edu/dataset) (Caltech), sites caltech, jpl et office001.

## Installation
1. `Rscript requirements.R`
2. Crée un fichier `.Renviron` à la racine contenant `ACN_API_TOKEN=<ton token>` (token gratuit sur ev.caltech.edu/register). Ce fichier est ignoré par git.
3. `Rscript src/data_acn.R` télécharge les sessions de septembre 2018 à mars 2020 (caltech, jpl, office001 : 46 701 sessions, environ 4 min) et les met en cache dans `data/`. Test : `Rscript tests/test_data_acn.R`.
4. Les notebooks sont rendus en R Markdown (`rmarkdown::render`), Quarto n'étant pas installé.

## Branches
- `projet-1-prediction-duree-energie` : durée et énergie prédites à l'arrivée
- `projet-2-profils-usagers` : profils d'usagers non supervisés
- `projet-3-fiabilite-depart` : départ anticipé par rapport à l'heure déclarée
- `synthese` : synthèse des trois pistes
