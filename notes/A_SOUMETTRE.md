# A soumettre — REVISE le 2026-08-08 apres le banc hors-echantillon

## ALERTE : ne PAS soumettre AA

`realpde_t1_AA.zip` a ete renomme `DANGER_NE_PAS_SOUMETTRE_realpde_t1_AA.zip.bak`.
Il supprimait nos bornes SPS sur la foi d'un raisonnement FAUX (voir plus bas). Mesure :
sans bornes, le facteur de bande tombe a ~20 contre nos 52.08, le sps passerait de 35.33
a ~13.5 et le final de 78.640 a **~72.9**. Une perte de pres de six points.

## A soumettre au prochain slot

| track | fichier | sha256 (16) | attendu |
|---|---|---|---|
| **T1** | `realpde_t1_Z.zip` | `3a85c13c9ed8f9cf` | 78.640 (deja mesure) |
| **T2** | `realpde_t2_F.zip` | `89081b6762d4f1f0` | >= 78.6 |

Aucun changement par rapport a ce qui tourne deja. Les deux idees que j'avais preparees
(retablir gamma, supprimer les bornes) sont l'une neutre, l'autre catastrophique.

> Le zip T2 qui avait fait 78.610 (`54e867600308d680`) a ete ecrase ; celui du disque a
> les tables PAR CANAL, jamais soumises, qui valaient +0.51 sps sur T1.

## L'ERREUR QUE LE BANC A RATTRAPEE

J'avais deduit du classement que l'entree `realpde` (C 54.75) etait le baseline sans
bornes, donc que nos tables (C 52.08) faisaient pire que rien. **C'etait faux.** Le banc
sur le vrai modele donne a la bande par defaut un C de 26.01 avec des residus
IN-SAMPLE — c'est-a-dire trop petits, donc un chiffre FLATTEUR. Elle ne peut pas valoir
54.75 en ligne ou les residus sont plus gros. Verification directe sur le classement :
les soumissions sans bornes sont a C 17-20 (`deleted_user_77183` : sps 13.54, C 19.86,
avec pourtant un bon modele). `realpde` rend donc ses propres bornes, et son tke de 79.30
n'est pas non plus celui du FNO nu : **ce n'est pas un baseline, c'est une soumission a
part entiere.** Toutes mes comparaisons contre elle etaient sans objet.

## L'ETAT REEL DU FRONT SPS

    sans bornes                        C ~ 20
    NOS BORNES                         C   52.08
    meilleur reel de la phase (yzf630) C   57.63
    anaelle_haomiao                    C   83.18  (anomalie : son final ne verifie pas la
                                                   formule, ligne de la charniere des phases)

Nos bornes valent donc **+32 points de C** par rapport a rien. Le gain encore disponible
est 52.08 -> 57.63, soit +3.8 sps et **+0.98 de final**. C'est le plus gros gisement
identifie et quantifie.

## CE QUE LE BANC HORS-ECHANTILLON A ETABLI

Instrument : FNO parti du pre-entrainement simulation seule, finetune 9172 pas sur 47
trajectoires, avec 35 trajectoires (aoa 15, aoa 20, Re 26761 — 42.7 % du cache) jamais
vues. Il plafonne a rl 92.5 sur ces regimes contre 96.3 pour `sim_real_ft` qui les a
memorises ; notre score en ligne (94.17) tombe entre les deux, ce qui valide l'instrument.

  * **lambda est un faux probleme.** La surface de C est PLATE : lambda 1.00 -> 2.00 ne
    deplace C que de 0.8, et notre 1.60 est a **99.7 % de l'oracle**. Raison structurelle :
    `hopt` est un argmax, il se recale tout seul quand on gonfle les residus. Le lambda
    deconfondu vaut 1.115 sur u et 1.518 sur v (et non 1.60 partout), mais le corriger ne
    rapporte que +0.06 de C.
  * **le centre de Wiener tient hors-echantillon** : +0.30 de C contre un centre sur la
    prediction, et l'optimum est meme plus bas (k=0.60). A garder.
  * **l'argmax bat tous les quantiles conformels** : C 43.44 contre 40.01 (q70), 36.31
    (q90), 31.75 (q95). La couverture optimale est **0.69**, pas 0.90. Viser 90 % coute
    7 points de C.
  * **gamma ne paie pas hors-echantillon** : le final est maximal a gamma 1.00 et decroit
    ensuite. La mesure en ligne disait +0.029 ; les deux sont dans le bruit. Ne pas
    toucher.
  * **la famille "table sur l'ecart-type d'entree" est presque epuisee** : in-sample, nous
    sommes a 60.45 contre 62.54 pour son oracle. Une table indexee par |pred| donne 60.84,
    soit un gain marginal.

## LE FRONT SPS EST EPUISE (mesure du 08/08 soir)

Neuf familles de bornes testees en LEAVE-ONE-REGIME-OUT (9 plis : 5 angles d'attaque,
4 bandes de Reynolds) sur le vrai modele. **La table expediee gagne partout.**

    enveloppe supremum inter-regimes   -2.56 C en moyenne, battue sur 8 plis/9,
                                       et perd AUSSI au pire cas (58.68 vs 59.85)
    equivariance d'echelle h = k.sd    -9.09 C
    hybride max(k.sd, table)           -6.13 C
    affine k.sd + h0                   -0.95 C
    puissance k.sd^p                   -0.51 C
    HDI dissymetrique                  +0.01 C
    quantiles conformels q70..q95      -7 a -12 C
    lambda, centre, bins fins, frames  a l'optimum ou nuls

Le covariate shift de regime est REEL et mesure (mediane du sd local x13.98 entre
regimes ; controle de bruit : enveloppe/mediane 1.36 sur partition physique contre 1.06
sur partition aleatoire). Mais l'elargissement n'est pas le remede : la penalite de
largeur est exponentielle et frappe TOUS les elements, alors que le gain de couverture ne
touche que la queue. Et la regle equivariante echoue parce que le residu n'est pas
proportionnel a la fluctuation locale -- il a un plancher irreductible.

**Les +0.98 vises ne sont donc PAS accessibles par la politique de bornes.** yzf630
obtient C 57.63 avec un modele moins bon que le notre (rl 93.89 contre 94.17) : le
mecanisme existe, mais neuf familles n'ont pas permis de le retrouver.

## OU CHERCHER LES +0.98 QUI RESTENT

Le seul pole ou l'ecart au peloton est MESURABLE et NON EXPLIQUE reste le **tke** :

    nous        74.03      tout le haut du classement : 76.0 a 79.6
    et pourtant notre rl (94.17) est MEILLEUR que celui de plusieurs d'entre eux

Le mecanisme est identifie : le FNO SOUS-DISPERSE, son rapport de fluctuation
pred/cible vaut 0.73 a 0.86 selon le regime, meme sur les trajectoires qu'il a vues.
Gamma global est neutre parce qu'il ajoute de la fluctuation MAL PLACEE partout, et paie
en rl ce qu'il gagne en tke.

**Piste non testee, seule restante avec un mecanisme physique clair** : un gamma
SPATIALEMENT VARIABLE, g(x,y) = sqrt(ke_entree / ke_pred), qui gonfle la ou la
sous-dispersion est reelle (le sillage) et ne touche a rien ailleurs. L'energie
cinetique de la FENETRE D'ENTREE est un predicteur legitime de celle de la sortie
(statistiques localement stationnaires) -- c'est d'ailleurs ce qui donnait au modele
statistique v15 un tke de 78.16, superieur a celui du FNO. Au bareme, on peut payer
0.2 de rl par point de tke gagne.

## CE QUI EST FERME PAR LA MESURE

- **LARG** (largeurs auto-calibrantes en ligne) : -7.58 sps en ligne.
- **Supprimer les bornes** : -32 points de C. Jamais.
- **Quantiles conformels a 90/95 %** : -7 a -12 points de C contre l'argmax.
- **lambda, le centre, gamma** : tous a l'optimum ou dans le bruit.
- **CNO, Transolver** : domines par le FNO et plus lents (x3.8, x8).
- **Empaquetage fp16 des poids** : ne coute rien (rl +0.000, tke +0.002).
- **Demi-precision a l'execution** : cuFFT la refuse (FFT sur 20, pas une puissance de 2).
- **`realpde` comme reference du FNO nu** : ce n'est pas un baseline.

## Rappels de plateforme

- Un slot par jour et par track, ouverture 02:00 locale.
- Une soumission FAILED ne consomme PAS le quota.
- 256 Mo extrait ; 5 min d'execution T1, 10 min T2.
- `local_eval.py` existe en v4 ET v6 dans le kit : toujours lancer explicitement le v6.
