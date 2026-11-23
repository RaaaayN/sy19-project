# PROGRESS : projet 3, fiabilité de l'heure de départ déclarée

- [x] 1. Données : heure déclarée vs réelle, taux de présence, choix et justification du seuil
- [x] 2. Cible binaire, déséquilibre des classes, découpage train/validation/test par date
- [x] 3. Baselines : classe majoritaire, taux historique de l'usager (passé seulement)
- [x] 4. Régression logistique, puis logistique L1 (glmnet, CV)
- [x] 5. LDA et QDA
- [x] 6. GAM logistique avec splines sur l'heure (mgcv)
- [x] 7. Arbre, bagging, random forest
- [x] 8. SVM linéaire et RBF (C, gamma par CV)
- [x] 9. MLP (torch ou nnet)
- [x] 10. Comparaison (AUC, F1, confusion, ROC, calibration) et investigation originale
- [x] 11. Module src/depart_risk.R + export CSV/JSON pour le simulateur Go (IA04)
- [x] 12. Rapport (reports/rapport.md), notebook reports/projet.Rmd rendu en HTML
- [ ] 13. Pull request vers main
