# Synthèse des trois projets SY19

Données : ACN-Data (Caltech), 46 701 sessions de recharge de sept. 2018 à mars 2020 (sites caltech, jpl, office001), chargées par `src/data_acn.R`. Tout le code est en R, avec un notebook R Markdown rendu en HTML par projet. Les chiffres ci-dessous sont repris des `reports/rapport.md` de chaque branche, eux-mêmes produits par les scripts commités.

## Tableau récapitulatif

| Projet | Branche | PR | Meilleur modèle | Métrique (test) | Baseline | Algos du cours couverts |
|---|---|---|---|---|---|---|
| 1. Durée et énergie à l'arrivée | `projet-1-prediction-duree-energie` | [#3](https://github.com/RaaaayN/sy19-project/pull/3) | MLP torch (retenu, exportable en Go) ; random forest équivalente | Durée : MAE 1,60 h, R² 0,55. Énergie : MAE 4,04 kWh, R² 0,70 | Moyenne par usager : MAE 2,09 h / 4,94 kWh, R² 0,36 / 0,54 | régression linéaire, ridge, lasso, sélection de variables, PCA (PCR), GAM et splines, arbre, bagging, random forest, SVR, réseau de neurones |
| 2. Profils d'usagers | `projet-2-profils-usagers` | [#2](https://github.com/RaaaayN/sy19-project/pull/2) | GMM K = 3, EM codé à la main (3 profils : pendulaire régulier, pendulaire flexible, visiteur irrégulier) | K = 3 par BIC et par vraisemblance sur usagers de test ; accord avec mclust ARI 0,907 ; profils expliqués par un arbre à 82,8 % | k-means k = 4 (silhouette 0,198 contre 0,080, plus stable) | PCA, k-means, GMM et EM, kernel k-means, LDA, arbre, one-class SVM |
| 3. Fiabilité de l'heure de départ déclarée | `projet-3-fiabilite-depart` | [#1](https://github.com/RaaaayN/sy19-project/pull/1) | MLP torch (AUC val. 0,852) ; random forest et GAM dans le même IC | AUC 0,865 [0,856 ; 0,875], F1 0,707 | Taux historique de l'usager : AUC 0,762 | logistique, lasso (L1), LDA, QDA, GAM logistique, arbre, bagging, random forest, SVM linéaire et RBF, réseau de neurones |

Investigations originales :
- P1 : les déclarations de l'usager n'aident que les nouveaux usagers ; l'historique domine.
- P2 : profils stables selon le seuil de sessions (5 à 20) et entre deux périodes (77 % des usagers gardent leur profil, kappa 0,64).
- P3 : les habitués sont plus fiables (36,5 % de départs anticipés à la 1re session, 24,8 % au-delà de 50) ; le rééquilibrage des classes n'améliore pas l'AUC et dégrade la calibration.

Sorties pour le simulateur IA04 (Go) : `reports/model_mlp.json` (P1), `reports/results/agents_gmm.json` (P2), `reports/export/depart_risk_gam.json` et tables CSV (P3).

## Couverture du programme SY19

| Méthode | P1 | P2 | P3 |
|---|---|---|---|
| Régression linéaire | x | | |
| LDA | | x | x |
| Régression logistique | | | x |
| Ridge | x | | |
| Lasso, sélection de variables | x | | x |
| PCA | x | x | |
| Splines et GAM | x | | x |
| Arbres | x | x | x |
| Bagging, random forest | x | | x |
| GMM et EM | | x | |
| SVM | | | x |
| SVR | x | | |
| One-class SVM | | x | |
| Kernel k-means | | x | |
| Réseaux de neurones | x | | x |

Tout le programme listé est couvert. Hors liste : QDA (P3) et k-means (P2). Non couverts : la sélection de variables pas à pas (stepwise, best subset), remplacée partout par le lasso, et la validation croisée temporelle (plis glissants), puisque les plis de CV sont aléatoires à l'intérieur du train (limite signalée en P1).

## Blocages et ce qu'il faut à l'humain

Aucun projet n'est bloqué. Il reste des décisions qui vous reviennent :

1. **Conformité au sujet.** Le sujet demande un seul projet par groupe, un notebook .Rmd/.qmd et son HTML, un rapport PDF de 6 pages au plus, le tout en un zip. Il faut choisir la piste principale, ou fusionner les trois en un seul notebook, puis écrire le PDF.
2. **Relire et merger les PR** #1, #2 et #3 (aucune n'est mergée).
3. **Points à défendre en soutenance :**
   - P3 : `requestedDeparture` est la dernière saisie de l'usager et peut contenir une information postérieure à l'arrivée.
   - P3 : la dérive temporelle rend les modèles sous-calibrés.
   - P2 : la structure en groupes est faible (silhouettes inférieures à 0,27) et k-means fait mieux sur ses propres critères.
   - P1 : la SVR est entraînée sur un sous-échantillon.
4. **Intégration IA04.** Les exports JSON/CSV n'ont pas encore été branchés dans le simulateur Go.
5. **Reproduction.** Chaque membre du groupe doit mettre son propre token ACN dans `.Renviron`, puis lancer `Rscript requirements.R` et `torch::install_torch()`. Le script random forest de P1 prend environ 25 min.
