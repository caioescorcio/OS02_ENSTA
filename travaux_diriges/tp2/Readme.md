# source ~/mpi-env/bin/activate

# TD n° 2 - 27 Janvier 2025

##  1. Parallélisation ensemble de Mandelbrot

L'ensensemble de Mandebrot est un ensemble fractal inventé par Benoit Mandelbrot permettant d'étudier la convergence ou la rapidité de divergence dans le plan complexe de la suite récursive suivante :
$$
\left\{
\begin{array}{l}
    c\,\,\textrm{valeurs\,\,complexe\,\,donnée}\\
    z_{0} = 0 \\
    z_{n+1} = z_{n}^{2} + c
\end{array}
\right.
$$
dépendant du paramètre $c$.

Il est facile de montrer que si il existe un $N$ tel que $\mid z_{N} \mid > 2$, alors la suite $z_{n}$ diverge. Cette propriété est très utile pour arrêter le calcul de la suite puisqu'on aura détecter que la suite a divergé. La rapidité de divergence est le plus petit $N$ trouvé pour la suite tel que $\mid z_{N} \mid > 2$.

On fixe un nombre d'itérations maximal $N_{\textrm{max}}$. Si jusqu'à cette itération, aucune valeur de $z_{N}$ ne dépasse en module 2, on considère que la suite converge.

L'ensemble de Mandelbrot sur le plan complexe est l'ensemble des valeurs de $c$ pour lesquels la suite converge.

Pour l'affichage de cette suite, on calcule une image de $W\times H$ pixels telle qu'à chaque pixel $(p_{i},p_{j})$, de l'espace image, on associe une valeur complexe  $c = x_{min} + p_{i}.\frac{x_{\textrm{max}}-x_{\textrm{min}}}{W} + i.\left(y_{\textrm{min}} + p_{j}.\frac{y_{\textrm{max}}-y_{\textrm{min}}}{H}\right)$. Pour chacune des valeurs $c$ associées à chaque pixel, on teste si la suite converge ou diverge.

- Si la suite converge, on affiche le pixel correspondant en noir
- Si la suite diverge, on affiche le pixel avec une couleur correspondant à la rapidité de divergence.

1. À partir du code séquentiel `mandelbrot.py`, faire une partition équitable par bloc suivant les lignes de l'image pour distribuer le calcul sur `nbp` processus  puis rassembler l'image sur le processus zéro pour la sauvegarder. Calculer le temps d'exécution pour différents nombre de tâches et calculer le speedup. Comment interpréter les résultats obtenus ?

```
4 processus   
Temps du calcul MOYEN de l'ensemble de Mandelbrot : 0.7514772415161133
Temps de constitution de l'image : 0.03166556358337402

1 processus
Temps du calcul de l'ensemble de Mandelbrot : 2.382941246032715
Temps de constitution de l'image : 0.049132585525512695
```

    Speed-up = 3,17x

    Le processus, quand divisé par différents tailles, fait que le calcul soit plus légere (moins d'opérations par processus). Donc il prend moins du temps

2. Réfléchissez à une meilleur répartition statique des lignes au vu de l'ensemble obtenu sur notre exemple et mettez la en œuvre. Calculer le temps d'exécution pour différents nombre de tâches et calculer le speedup et comparez avec l'ancienne répartition. Quel problème pourrait se poser avec une telle stratégie ?

```
processus: 2
Temps du calcul MOYEN de l'ensemble de Mandelbrot : 1.7081185579299927
Temps de constitution de l'image : 0.03320574760437012

processus: 3      
Temps du calcul MOYEN de l'ensemble de Mandelbrot : 1.0858351389567058
Temps de constitution de l'image : 0.026440143585205078

processus: 4      
Temps du calcul MOYEN de l'ensemble de Mandelbrot : 0.7926850914955139
Temps de constitution de l'image : 0.027698278427124023

processus: 5      
Temps du calcul MOYEN de l'ensemble de Mandelbrot : 0.6393367767333984
Temps de constitution de l'image : 0.04821491241455078

processus: 6      
Temps du calcul MOYEN de l'ensemble de Mandelbrot : 0.6458926995595297
Temps de constitution de l'image : 0.026892423629760742
```

    Il semble que il y a une constraint de temps pour le nombre de processus, ça peut être un resultat de la borne d'optimisation du programme. Donc, il peut exister un temps minimal pour l'execution des données qui ne dépend pas du nombre de tâches.

3. Mettre en œuvre une stratégie maître-esclave pour distribuer les différentes lignes de l'image à calculer. Calculer le speedup avec cette approche et comparez  avec les solutions différentes. Qu'en concluez-vous ?

```
processus 2
time:  2.4339230060577393

processus 3 
time:  1.125908374786377

processus 4 
time:  0.8213310241699219

processus 5 
time:  0.7039659023284912

processus 6 
time:  0.5637714862823486
```

    Il semble avoir plus du potentiel pour accelerer le temps d'éxecution lorsqu'il fait une division par disponibilité, pas forcement une division d'ordre de rang. Alors, même avec moins un processus (rang 0 ne travaille pas comme les autres) il peut avoir une amélioration par rapport à l'autre méthode

## 2. Produit matrice-vecteur

On considère le produit d'une matrice carrée $A$ de dimension $N$ par un vecteur $u$ de même dimension dans $\mathbb{R}$. La matrice est constituée des cœfficients définis par $A_{ij} = (i+j) \mod N$. 

Par soucis de simplification, on supposera $N$ divisible par le nombre de tâches `nbp` exécutées.

### a - Produit parallèle matrice-vecteur par colonne

Afin de paralléliser le produit matrice–vecteur, on décide dans un premier temps de partitionner la matrice par un découpage par bloc de colonnes. Chaque tâche contiendra $N_{\textrm{loc}}$ colonnes de la matrice. 

- Calculer en fonction du nombre de tâches la valeur de Nloc
- Paralléliser le code séquentiel `matvec.py` en veillant à ce que chaque tâche n’assemble que la partie de la matrice utile à sa somme partielle du produit matrice-vecteur. On s’assurera que toutes les tâches à la fin du programme contiennent le vecteur résultat complet.
- Calculer le speed-up obtenu avec une telle approche

```bash
Temps séquentiel: 0.0026133060455322266
Temps parallèle: 0.00022220611572265625
Speedup =  11.760729613733906
```


### b - Produit parallèle matrice-vecteur par ligne

Afin de paralléliser le produit matrice–vecteur, on décide dans un deuxième temps de partitionner la matrice par un découpage par bloc de lignes. Chaque tâche contiendra $N_{\textrm{loc}}$ lignes de la matrice.

- Calculer en fonction du nombre de tâches la valeur de Nloc
- paralléliser le code séquentiel `matvec.py` en veillant à ce que chaque tâche n’assemble que la partie de la matrice utile à son produit matrice-vecteur partiel. On s’assurera que toutes les tâches à la fin du programme contiennent le vecteur résultat complet.
- Calculer le speed-up obtenu avec une telle approche

```bash
Temps séquentiel: 0.0034008026123046875
Temps parallèle: 0.000240325927734375
Speedup =  14.15079365079365
```

## 3. Entraînement pour l'examen écrit

Alice a parallélisé en partie un code sur machine à mémoire distribuée. Pour un jeu de données spécifiques, elle remarque que la partie qu’elle exécute en parallèle représente en temps de traitement 90% du temps d’exécution du programme en séquentiel.

En utilisant la loi d’Amdhal, pouvez-vous prédire l’accélération maximale que pourra obtenir Alice avec son code (en considérant n ≫ 1) ? 

    S(n)=1/((1−P)+P/n);

    n >> 1 => Smax = 1/0.1 = 10

À votre avis, pour ce jeu de donné spécifique, quel nombre de nœuds de calcul semble-t-il raisonnable de prendre pour ne pas trop gaspiller de ressources CPU ?

    Ne pas utiliser beaucoup plus de nœuds que l’accélération maximale théorique. Donc, ~10.

En effectuant son cacul sur son calculateur, Alice s’aperçoit qu’elle obtient une accélération maximale de quatre en augmentant le nombre de nœuds de calcul pour son jeu spécifique de données.

En doublant la quantité de donnée à traiter, et en supposant la complexité de l’algorithme parallèle linéaire, quelle accélération maximale peut espérer Alice en utilisant la loi de Gustafson ?

    n = 4, car l'amélioration observé est égal à 4

    Sg​(n)=n−(1−P)(n−1), P = 0.9, n = 4

    Sg(4) = 3.7

    Pour une quantité doublé de données, en supposant que le problème reste lineaire: Sg = 3,7*2 = 7.4, peut être attendue.

