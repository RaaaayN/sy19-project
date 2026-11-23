# Projet 3 : peut-on se fier à l'heure de départ déclarée ?

SY19, UTC. Données ACN-Data (Caltech), sessions de recharge de véhicules électriques de septembre 2018 à février 2020.
Tout le code est en R ; chaque chiffre de ce rapport sort d'un script de `src/` et d'un fichier de `reports/metrics/`
(indiqué entre parenthèses). Le notebook `reports/projet.Rmd` (rendu `reports/projet.html`) relit ces fichiers.

## 1. Question et lien avec IA04

À son arrivée, l'usager déclare dans l'application ACN l'heure à laquelle il compte partir (`requestedDeparture`).
Le simulateur multi-agents d'IA04 (Go, puissance de site limitée, stratégies EDF, LLF, sLLF) utilise cette heure
comme échéance : la laxité LLF vaut (échéance - maintenant) - énergie restante / puissance max. Quand l'usager
part plus tôt que prévu, la laxité était surestimée et la voiture repart moins chargée que promis. On veut donc
prédire, dès l'arrivée, la probabilité qu'un usager parte **plus de 15 minutes avant l'heure déclarée**,
pour décider quand le coordinateur peut faire confiance à cette heure et sinon raccourcir l'échéance.

## 2. Données et cible (`src/p3_data.R`)

**Présence de l'heure déclarée** (`p3_presence.csv`). Elle n'existe que pour les sessions avec un `userID`
(une saisie dans l'application) : 66,9 % des sessions à caltech, 92,5 % à jpl, 7,8 % à office001.
L'heure déclarée est donc disponible assez souvent pour définir la cible directement ; la cible de repli
(« départ avant la médiane de l'usager ») n'a pas été nécessaire. Le biais de sélection est réel : les sessions
sans compte (visiteurs, paiement direct) sont absentes de l'étude, et le modèle ne s'applique qu'aux usagers
qui déclarent une heure, ce qui est justement le cas où le coordinateur en dispose.

**Filtres.** On garde caltech et jpl (office001 : 93 sessions avec heure déclarée, toutes en 2020, donc rien
pour l'apprentissage), les durées déclarées dans ]0, 24 h] et les durées réelles dans ]0, 48 h], et on s'arrête
au 29 février 2020 (mars 2020 : début du confinement, comportement atypique). Il reste **35 930 sessions et
736 usagers**. Les heures sont converties de GMT vers America/Los_Angeles avant tout calcul.

**Seuil** (`p3_seuils.csv`, figure `p3_ecart_depart.png`). Écart = départ réel - départ déclaré. La part des
sessions parties avant l'heure déclarée dépend du seuil de tolérance :

| seuil (min) | 0 | 15 | 30 | 60 | 120 |
|---|---|---|---|---|---|
| taux de départ anticipé | 33,8 % | 27,4 % | 23,3 % | 17,5 % | 11,2 % |

On retient 15 min : (i) un départ à quelques minutes près n'a pas d'effet sur le planning (le pas des
ordonnanceurs ACN et du simulateur est de quelques minutes, et à puissance typique 15 min représentent peu
d'énergie) ; (ii) le seuil 0 compterait comme « non fiables » des usagers qui partent à l'heure à la minute près,
les déclarations étant arrondies ; (iii) au-delà (30, 60 min) la classe positive se réduit sans changer la
nature du problème. Le seuil est un paramètre (`P3_SEUIL`) et le tableau ci-dessus permet d'en juger l'effet.

**Variables, toutes connues à l'arrivée.** Site, heure d'arrivée, heure de départ déclarée, durée déclarée
(`minutesAvailable`), week-end, `kWhRequested`, `WhPerMile`, et l'historique de l'usager : nombre de sessions
passées (`n_past`), taux lissé de départs anticipés passés (`past_rate`) et écart moyen passé (`past_gap`, h).
L'historique n'utilise que les sessions **terminées** (déconnexion antérieure à l'arrivée courante), ce que
vérifie `tests/test_p3_data.R` sur 200 sessions tirées au hasard. Le taux est lissé vers le taux moyen du train
p0 avec un poids de 5 sessions : (k + 5 p0) / (n + 5). `disconnectTime`, `doneChargingTime` et `kWhDelivered`
sont exclus (connus après le départ).

**Découpage par date et déséquilibre** (`p3_classes.csv`, figure `p3_taux_mensuel.png`) :

| jeu | période | sessions | usagers | taux de départ anticipé |
|---|---|---|---|---|
| train | sept. 2018 - août 2019 | 23 539 | 606 | 25,4 % |
| validation | sept. - nov. 2019 | 6 393 | 445 | 28,6 % |
| test | déc. 2019 - fév. 2020 | 5 998 | 437 | 33,7 % |

La classe positive représente un quart à un tiers des sessions : déséquilibre modéré, mais qui suffit à rendre
l'exactitude trompeuse (prédire toujours « fiable » donne 66 % sur le test). On évalue donc par l'AUC (insensible
au seuil et à la prévalence), le F1 de la classe « départ anticipé », le score de Brier et la calibration.
Le taux augmente au cours du temps (dérive), ce qui pèse sur la calibration (section 5).

## 3. Protocole (`src/p3_eval.R`)

- Hyperparamètres choisis **sur le train seulement** : CV 10 plis (lasso, arbre), AUC out-of-bag (forêts),
  CV 3 plis (SVM), hold-out des 20 % les plus récents du train (MLP), REML (GAM).
- Variables standardisées avec les moyennes et écarts-types du train.
- Le seuil de décision de chaque modèle est celui qui maximise le F1 **sur la validation** ; on l'applique tel quel
  au test.
- Le modèle est sélectionné sur l'AUC de validation ; le test ne sert qu'à rapporter les performances
  (avec un intervalle de confiance DeLong à 95 %).

## 4. Modèles

Chaque section : pourquoi, résultat (validation / test), interprétation. Le tableau complet est en section 5.

### 4.1 Baselines (`src/p3_baselines.R`)
*Classe majoritaire* : prédit toujours « fiable » ; AUC 0,5 et F1 nul, Brier test 0,230.
*Taux historique de l'usager* (`past_rate`, passé seulement) : c'est la règle qu'appliquerait un coordinateur sans
modèle. AUC 0,747 / 0,762, F1 0,543 / 0,606. Le passé de l'usager est déjà très informatif ; tout modèle doit
faire mieux que cette référence.

### 4.2 Régression logistique et lasso (`src/p3_logit.R`)
Modèle linéaire de référence, interprétable (`p3_logit_coefs.csv`). AUC 0,820 / 0,832, F1 0,622 / 0,666.
Sur variables standardisées, les effets les plus forts sont `past_rate` (+0,70), la durée déclarée (+0,61),
l'heure d'arrivée (+0,57) et l'heure de départ déclarée (+0,27) : plus la promesse est longue et tardive, plus
le départ anticipé est probable. Le lasso (λ choisi par CV 10 plis, règle 1-SE, λ = 0,0122, figures
`p3_lasso_cv.png`, `p3_lasso_chemin.png`) garde **6 variables sur 10** et élimine le site, `kWhRequested`,
`WhPerMile` et `past_gap` (`p3_lasso_coefs.csv`), pour une AUC quasi identique (0,818 / 0,830) : ces variables
n'apportent rien une fois l'historique et les horaires connus.

### 4.3 LDA et QDA (`src/p3_da.R`)
La LDA (covariance commune) donne une frontière linéaire : AUC 0,818 / 0,830, comme la logistique, ce qui est
attendu. La QDA (une covariance par classe) se dégrade : AUC 0,794 / 0,811 et surtout Brier 0,201, la plus mauvaise
calibration (figure `p3_calibration_test.png`) ; ses probabilités sont écrasées vers 0 et 1 car l'hypothèse
gaussienne est fausse pour des variables binaires, bornées (taux) ou très asymétriques (`n_past`).

### 4.4 GAM logistique (`src/p3_gam.R`)
Les heures ont un effet évidemment non linéaire et cyclique : on utilise des splines cycliques (23 h 59 voisin de
0 h) pour l'heure d'arrivée et l'heure déclarée, des splines pénalisées pour la durée et l'historique, lissage
par REML et `select = TRUE` (un terme inutile peut être annulé). AUC 0,845 / 0,857, F1 0,651 / 0,702, Brier 0,144.
Tous les termes lisses sont retenus (edf de 2,9 à 10,6, `p3_gam_termes.csv`). Lecture des effets
(`p3_gam_effets.png`) : un départ déclaré en soirée ou la nuit (après 20 h, avant 5 h) est très peu fiable ;
le risque croît avec la durée déclarée jusque vers 18 h ; il croît presque linéairement avec `past_rate` ;
il est plus élevé pour les nouveaux usagers. Le GAM gagne 0,025 d'AUC sur la logistique, seulement en
assouplissant la forme des effets. Il est additif, donc exportable sous forme de tables (section 7).

### 4.5 Arbre, bagging, random forest (`src/p3_trees.R`)
Arbre CART élagué par CV 10 plis (règle 1-SE, cp = 0,00125, 29 feuilles, `p3_arbre.png`) : lisible mais
instable et moins bon, AUC 0,801 / 0,811. Bagging (forêt avec mtry = 10, nodesize = 20 retenu par AUC OOB) :
0,836 / 0,848. Random forest (grille mtry × nodesize, `p3_rf_grille.csv` ; retenu mtry = 2, nodesize = 1,
AUC OOB 0,841) : **0,848 / 0,861**, F1 0,674 / 0,707 avec la meilleure précision (0,751) et le meilleur Brier
(0,138). La décorrélation des arbres (mtry petit) apporte nettement plus que le bagging seul. Importance
(`p3_rf_importance.png`) : heure de départ déclarée et `past_rate` en tête, puis durée déclarée et `past_gap`.

### 4.6 SVM linéaire et RBF (`src/p3_svm.R`)
Pour libsvm (coût au moins quadratique en n), la grille est évaluée en CV 3 plis sur **8 000 sessions tirées
du train** ; le modèle final (probabilités de Platt) est ajusté sur tout le train. Score de CV : AUC des valeurs
de décision. Linéaire : C indifférent entre 0,01 et 10 (AUC CV 0,786-0,787), C = 10 retenu, AUC 0,820 / 0,833 :
même chose que la logistique. RBF : grille C ∈ {0,3 ; 1 ; 3 ; 10 ; 30} × γ ∈ {0,003 ; … ; 0,3}, retenu C = 3,
γ = 0,003 (AUC CV 0,793), AUC 0,810 / 0,826. Le noyau RBF ne bat pas le linéaire en AUC (meilleur F1, 0,692) :
le γ retenu est très petit, c'est-à-dire un noyau presque linéaire, et la frontière utile (créneaux horaires)
est mieux captée par les splines ou les arbres. libsvm signale des arrêts au nombre maximal d'itérations pour
quelques C élevés, ce qui est une limite de cette partie.

### 4.7 MLP (`src/p3_mlp.R`, paquet R `torch`)
Deux couches cachées ReLU avec dropout 0,2, Adam (lr 1e-3), entropie croisée. La taille (16 ou 64) et le weight
decay (0 ou 1e-3) sont choisis sur le hold-out formé par les 20 % les plus récents du train, ainsi que le nombre
d'époques (`p3_mlp_grille.csv`, `p3_mlp_courbes.png`) ; retenu 64 neurones, sans weight decay, 96 époques
(AUC hold-out 0,835), puis réajustement sur tout le train. **AUC 0,852 / 0,865**, F1 0,666 / 0,707, Brier 0,140.
C'est le meilleur en AUC, de peu : son IC test [0,856 ; 0,875] recouvre celui de la forêt et du GAM.

## 5. Comparaison (`src/p3_compare.R`)

Trié par AUC de validation (`p3_comparaison.csv`). F1, précision et rappel au seuil choisi sur la validation.

| modèle | AUC val | AUC test [IC 95 %] | F1 val | F1 test | précision test | rappel test | Brier test |
|---|---|---|---|---|---|---|---|
| **MLP (torch)** | **0,852** | **0,865** [0,856 ; 0,875] | 0,666 | 0,707 | 0,692 | 0,724 | 0,140 |
| random forest | 0,848 | 0,861 [0,851 ; 0,871] | 0,674 | 0,707 | 0,751 | 0,667 | 0,138 |
| GAM | 0,845 | 0,857 [0,847 ; 0,867] | 0,651 | 0,702 | 0,661 | 0,748 | 0,144 |
| bagging | 0,836 | 0,848 [0,837 ; 0,858] | 0,662 | 0,695 | 0,675 | 0,717 | 0,142 |
| logistique | 0,820 | 0,832 [0,822 ; 0,843] | 0,622 | 0,666 | 0,626 | 0,713 | 0,162 |
| SVM linéaire | 0,820 | 0,833 [0,822 ; 0,843] | 0,623 | 0,668 | 0,639 | 0,699 | 0,161 |
| LDA | 0,818 | 0,830 [0,819 ; 0,840] | 0,619 | 0,670 | 0,605 | 0,751 | 0,163 |
| lasso | 0,818 | 0,830 [0,819 ; 0,840] | 0,620 | 0,665 | 0,620 | 0,717 | 0,164 |
| SVM RBF | 0,810 | 0,826 [0,814 ; 0,838] | 0,639 | 0,692 | 0,697 | 0,687 | 0,159 |
| arbre | 0,801 | 0,811 [0,800 ; 0,823] | 0,614 | 0,660 | 0,565 | 0,794 | 0,157 |
| QDA | 0,794 | 0,811 [0,800 ; 0,822] | 0,606 | 0,649 | 0,589 | 0,722 | 0,201 |
| taux historique usager | 0,747 | 0,762 [0,750 ; 0,775] | 0,543 | 0,606 | 0,522 | 0,722 | 0,184 |
| classe majoritaire | 0,500 | 0,500 | 0 | 0 | 0 | 0 | 0,230 |

**Matrice de confusion du MLP sur le test** (`p3_confusion_test.csv`, seuil 0,291 issu de la validation) :

| | prédit anticipé | prédit fiable |
|---|---|---|
| **parti > 15 min en avance** | 1 464 | 559 |
| **parti à l'heure ou après** | 653 | 3 322 |

Lecture :
- Trois familles se détachent : modèles flexibles (MLP, forêt, GAM : AUC test 0,857-0,865), modèles linéaires
  (logistique, lasso, LDA, SVM linéaire : 0,830-0,833) et modèles fragiles (arbre seul, QDA : 0,811).
  Le gain du MLP sur la baseline « taux de l'usager » est significatif (DeLong, p ≈ 2e-80, `p3_selection.csv`).
- Les performances test sont un peu meilleures que celles de validation pour tous les modèles : le test a plus
  de positifs (33,7 %), ce qui rend la classe plus facile à rattraper en F1 ; l'AUC suit le même sens.
- **Calibration** (`p3_calibration_test.png`) : tous les modèles sous-estiment le risque sur le test car le taux
  de départ anticipé est passé de 25,4 % (train) à 33,7 % (test). Pour un usage dans le simulateur, il faut donc
  recalibrer périodiquement (par exemple réajuster l'intercept sur les dernières semaines). La QDA est de loin la
  plus mal calibrée.
- Courbes ROC : `p3_roc_test.png` ; AUC avec IC : `p3_auc_modeles.png`.

**Choix.** Le MLP est retenu sur l'AUC de validation, mais l'écart avec la forêt et le GAM est dans le bruit.
Pour l'export vers Go on retient **le GAM** : AUC test 0,857 (0,858 réajusté sur train + validation), à 0,008 du
MLP, et additif, donc transportable sous forme de quelques tables sans dépendance (section 7).

## 6. Investigation originale (`src/p3_investigation.R`)

### 6.1 La fiabilité évolue-t-elle avec l'ancienneté de l'usager ?
(`p3_invest_anciennete.csv`, `p3_invest_rang_intra_usager.csv`, figure `p3_invest_anciennete.png`)

| sessions passées | sessions | taux de départ anticipé | AUC test MLP | AUC test taux usager |
|---|---|---|---|---|
| 0 | 737 | 36,5 % | 0,687 | (constant) |
| 1-4 | 2 366 | 35,5 % | 0,790 | 0,678 |
| 5-19 | 6 407 | 29,7 % | 0,838 | 0,741 |
| 20-49 | 8 497 | 27,9 % | 0,858 | 0,727 |
| 50+ | 17 923 | 24,8 % | 0,871 | 0,775 |

1. **Les habitués sont plus fiables** : le taux de départ anticipé baisse de 36,5 % (première session) à 24,8 %
   (au-delà de 50). L'effet existe aussi **à l'intérieur d'un même usager** (usagers ayant au moins 40 sessions) :
   31,2 % sur ses 5 premières sessions, 26,7 % sur les sessions 11 à 20, 25,0 % au-delà de la 41e. Une partie de
   l'effet vient donc de l'apprentissage de l'application et de ses habitudes, pas seulement d'une sélection des
   usagers fidèles.
2. **Le modèle est d'autant plus utile que l'historique est long**, mais il apporte surtout là où le taux seul
   échoue : pour les nouveaux usagers (aucun historique), le MLP atteint quand même 0,687 grâce aux horaires et à la
   durée déclarée, et pour 1 à 4 sessions il gagne 0,11 d'AUC sur le taux historique. Pour le coordinateur, la
   conséquence est simple : l'heure d'un nouvel usager est la moins fiable et la plus difficile à prédire, il faut
   lui appliquer la marge par défaut.

### 6.2 Effet du rééquilibrage des classes
(`p3_invest_reequilibrage.csv`) Logistique et forêt (mtry = 2) entraînées sans rééquilibrage, avec poids
inverses aux fréquences, ou par sous-échantillonnage de la classe majoritaire.

| variante | AUC test | F1 test (seuil validation) | F1 test (seuil 0,5) | proba moyenne test | Brier |
|---|---|---|---|---|---|
| logistique, aucun | 0,832 | 0,666 | 0,526 | 0,267 | 0,162 |
| logistique, poids | 0,832 | 0,676 | 0,666 | 0,441 | 0,166 |
| logistique, sous-échantillonnage | 0,831 | 0,672 | 0,665 | 0,440 | 0,166 |
| forêt, aucun | 0,858 | 0,707 | 0,679 | 0,326 | 0,139 |
| forêt, sous-échantillonnage | 0,864 | 0,714 | 0,717 | 0,418 | 0,144 |

(taux observé sur le test : 33,7 %). Le rééquilibrage **ne change presque pas l'AUC** (classement identique) ;
il ne fait que déplacer les probabilités vers 0,5. Son seul effet visible, un meilleur F1 au seuil naïf 0,5, est
obtenu aussi bien en choisissant le seuil sur la validation sans rééquilibrer. En contrepartie il **dégrade la
calibration** (Brier plus élevé, probabilité moyenne 0,44 pour une fréquence réelle de 0,34). Comme le simulateur
utilise la probabilité elle-même pour dimensionner une marge, on ne rééquilibre pas. (La forêt sous-échantillonnée
gagne 0,006 d'AUC, dans l'IC.)

## 7. Module de risque et export pour Go (`src/p3_export.R`, `src/depart_risk.R`)

- Le GAM est réajusté sur train + validation (mêmes réglages REML) : AUC test 0,858.
- `reports/export/depart_risk_gam.json` : intercept, coefficients linéaires (site, week-end, kWh, Wh/mile) et,
  pour chaque terme lisse, une grille et ses valeurs (pas de 15 min pour les heures). En Go :
  `logit = intercept + Σ linéaires + Σ interp(grille, valeur, x)` (interpolation linéaire, valeur extrême hors
  grille), puis `p = 1 / (1 + exp(-logit))`. Le fichier contient aussi le prior (p0 = 0,261, m = 5) pour calculer
  `past_rate`, des valeurs par défaut, et l'avance médiane quand l'usager part trop tôt (89,6 min ; 3e quartile
  191,8 min).
- `reports/export/depart_risk_usagers.csv` : état de l'historique de chaque usager à la fin des données (n, k,
  écart moyen, taux lissé) pour initialiser les agents.
- `reports/export/depart_risk_creneaux.csv` : probabilité moyenne prédite et taux observé par site × créneau de
  3 h × ancienneté (test), table de repli si l'on ne veut pas calculer le GAM.
- `depart_risk()` (R) fait exactement le calcul que fera Go à partir du JSON ; `tests/test_depart_risk.R` vérifie
  l'écart avec `predict.gam` (< 0,01, dû à l'interpolation), des probabilités dans ]0, 1[ et la croissance du
  risque avec le taux historique.
- **Usage dans LLF** : `marge_llf(p) = p × 89,6 min`, retranchée de l'échéance déclarée :
  `échéance_effective = requestedDeparture - marge_llf(p)`. Une voiture dont l'usager est peu fiable voit sa laxité
  diminuer et passe plus tôt dans la file. Une variante plus prudente utilise le 3e quartile de l'avance, ou une
  règle à seuil (marge seulement si p > 0,29, seuil F1 du MLP).

## 8. Limites

- `requestedDeparture` est la **dernière** saisie de l'usager (`src/data_acn.R`) ; si l'usager modifie son heure
  pendant la session, une information postérieure à l'arrivée peut s'y glisser. Les données exposées ne donnent pas
  l'horodatage des modifications, on ne peut pas mesurer ce biais.
- Dérive : le taux de départ anticipé croît dans le temps ; modèles sous-calibrés sur le test. À recalibrer.
- Sélection : seules les sessions avec compte usager sont étudiées ; deux sites seulement (office001 écarté).
- SVM ajustés sur un sous-échantillon pour la CV, avertissements de convergence de libsvm pour certains C.
- Le MLP est encore en légère amélioration à la dernière époque testée (100) ; un budget plus grand gagnerait
  peut-être un peu, sans changer le classement global vu les IC.
- La marge LLF proposée (p × avance médiane) n'a pas été testée dans le simulateur ; c'est l'étape suivante côté IA04
  (comparer énergie non livrée avec et sans marge).

## 9. Reproduire

```
Rscript requirements.R                      # puis torch::install_torch() une fois
Rscript src/p3_data.R && Rscript tests/test_p3_data.R
Rscript src/p3_baselines.R; Rscript src/p3_logit.R; Rscript src/p3_da.R; Rscript src/p3_gam.R
Rscript src/p3_trees.R; Rscript src/p3_svm.R; Rscript src/p3_mlp.R
Rscript src/p3_compare.R; Rscript src/p3_investigation.R
Rscript src/p3_export.R && Rscript tests/test_depart_risk.R
Rscript -e 'rmarkdown::render("reports/projet.Rmd")'
```
