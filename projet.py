#Iarina NISTOR 21210925

#Les réponses aux questions se trouvent à la fin de ce fichier.


import numpy as np
import utils as u
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
train=pd.read_csv("train.csv")


def getPrior(data):
    """
    Calcule une estimation préalable (prior) de la variable cible, 
    ainsi qu'un intervalle de confiance à 95 % (±1,96 erreurs standard).

    Args:
        data (dict) : Un dictionnaire contenant les données, avec une clé 'target' 

    Returns:
        dict : Un dictionnaire contenant les éléments suivants :
            - 'estimation' : La moyenne des valeurs de la variable cible.
            - 'min5pourcent' : La borne inférieure de l'intervalle de confiance à 95 %.
            - 'max5pourcent' : La borne supérieure de l'intervalle de confiance à 95 %.
    """
    # Extraction de la variable cible sous forme de tableau numpy
    target = data['target'].to_numpy()
    
    # Calcul de la moyenne de la variable cible
    estimation = np.mean(target)
    
    # Calcul de l'écart-type de la variable cible
    ecart_type = np.std(target)
    
    # Nombre de données dans la variable cible
    cpt = len(target)
    
    # Calcul des bornes de l'intervalle de confiance à 95 %
    min5pourcent = estimation - 1.96 * ecart_type / np.sqrt(cpt)
    max5pourcent = estimation + 1.96 * ecart_type / np.sqrt(cpt)
    
    # Retourne les résultats sous forme de dictionnaire
    return {
        'estimation': estimation,
        'min5pourcent': min5pourcent,
        'max5pourcent': max5pourcent
    }



def P2D_l(df, attr):
    """
    Calcule dans le dataframe la probabilité conditionnelle P(attr|target), 
    et la retourne sous forme d'un dictionnaire imbriqué. 
    Pour chaque valeur de 'target' (t), un dictionnaire associe à chaque valeur 
    possible de 'attr' (a) la probabilité conditionnelle P(attr=a|target=t).

    Args:
        df (DataFrame) : Le dataframe contenant les données à analyser.
        attr (str) : Le nom de la colonne représentant l'attribut pour lequel 
                     on souhaite calculer la probabilité conditionnelle.

    Returns:
        dict : Un dictionnaire structuré comme suit :
            - clé : valeur t de la colonne 'target'.
            - valeur : un dictionnaire avec :
                - clé : valeur a de l'attribut 'attr'.
                - valeur : probabilité conditionnelle P(attr=a|target=t).
    """
    # Utilisation de pd.crosstab pour calculer les fréquences conditionnelles
    # normalize='columns' normalise les valeurs pour obtenir P(attr|target)
    return pd.crosstab(df[attr], df['target'], normalize='columns').to_dict()  # P(attr|target)


def P2D_p(df, attr):
    """
    Calcule dans le dataframe la probabilité conditionnelle P(target|attr), 
    et la retourne sous forme d'un dictionnaire imbriqué. 
    Pour chaque valeur de 'attr' (a), un dictionnaire associe à chaque valeur 
    possible de 'target' (t) la probabilité conditionnelle P(target=t|attr=a).

    Args:
        df (DataFrame) : Le dataframe contenant les données à analyser.
        attr (str) : Le nom de la colonne représentant l'attribut pour lequel 
                     on souhaite calculer la probabilité conditionnelle.

    Returns:
        dict : Un dictionnaire structuré comme suit :
            - clé : valeur a de la colonne 'attr'.
            - valeur : un dictionnaire avec :
                - clé : valeur t de la colonne 'target'.
                - valeur : probabilité conditionnelle P(target=t|attr=a).
    """
    # Utilisation de pd.crosstab pour calculer les fréquences conditionnelles
    # normalize='columns' normalise les valeurs pour obtenir P(target|attr)
    return pd.crosstab(df['target'], df[attr], normalize='columns').to_dict()  # P(target|attr)



def decompose_taille(taille_octets):
    """
    Décompose une taille en octets en une représentation plus lisible, 
    avec des unités hiérarchiques (Go, Mo, Ko, o).

    Args:
        taille_octets (int) : La taille en octets à convertir.

    Returns:
        str : Une chaîne de caractères représentant la taille décomposée 
              en gigaoctets (Go), mégaoctets (Mo), kilo-octets (Ko) et octets (o),
              selon le cas.
    """
    # Chaîne qui stockera la représentation finale
    aff = ''
    
    # Initialisation de la taille en octets
    s = taille_octets
    
    # Vérifie si la taille est supérieure à 1 Ko
    if s > 1024:
        aff += "= "
        
        # Conversion des octets en kilo-octets (Ko)
        ko = int(s / 1024)
        s = s % 1024
        
        # Vérifie si la taille dépasse 1 Mo
        if ko > 1024:
            mo = int(ko / 1024)
            ko = ko % 1024
            
            # Vérifie si la taille dépasse 1 Go
            if mo > 1024:
                go = int(mo / 1024)
                mo = mo % 1024
                aff += str(go) + "go "  # Ajoute les gigaoctets à la chaîne finale
            
            # Ajoute les mégaoctets à la chaîne finale
            aff += str(mo) + "mo "
        
        # Ajoute les kilo-octets à la chaîne finale
        aff += str(ko) + "ko "
    
    # Ajoute les octets restants à la chaîne finale
    aff += str(s) + "o"
    
    return aff

            
def nbParams(df, attrs=[]):
    """
    Calcule la taille mémoire nécessaire pour stocker les paramètres uniques 
    dans un ensemble d'attributs spécifiés ou dans l'ensemble des colonnes d'un dataframe, 
    puis affiche cette taille sous une forme lisible.

    Args:
        df (DataFrame) : Le dataframe contenant les données.
        attrs (list, optional) : Une liste des noms des attributs (colonnes) 
                                 pour lesquels calculer le nombre de paramètres. 
                                 Si la liste est vide, la taille sera calculée 
                                 pour toutes les colonnes du dataframe. 
                                 Par défaut, `attrs=[]`.

    Returns:
        int : La taille mémoire en octets pour stocker les paramètres uniques.
    """
    # Initialisation de la longueur : nombre de variables analysées
    if attrs == []:
        longueur = 14  # Valeur par défaut
    else:
        longueur = len(attrs)  # Nombre d'attributs spécifiés

    # Taille de base : 8 octets par élément (hypothèse : float)
    taille_octets = 8 

    # Calcul de la taille mémoire en fonction des attributs spécifiés
    for att in attrs:
        # Multiplication par le nombre de valeurs uniques pour chaque attribut
        taille_octets *= len(np.unique(df[att]))
    
    # Si aucun attribut n'est spécifié, calcul pour toutes les colonnes du dataframe
    if attrs == []:
        for _, col_data in df.items():
            taille_octets *= len(np.unique(col_data))

    # Conversion de la taille en une représentation lisible
    aff = decompose_taille(taille_octets)
    
    # Affichage des informations sur la taille
    print(f'{longueur} variable(s) : {taille_octets} octets {aff}')
    
    # Retourne la taille totale en octets
    return taille_octets

def nbParamsIndep(df):
    """
    Calcule la taille mémoire nécessaire pour stocker les paramètres uniques 
    de manière indépendante pour chaque colonne (attribut) du dataframe, 
    puis affiche cette taille sous une forme lisible.

    Args:
        df (DataFrame) : Le dataframe contenant les données.

    Returns:
        None : Cette fonction ne retourne rien, mais affiche :
            - Le nombre de colonnes (variables).
            - La taille totale en octets pour stocker les paramètres uniques.
            - Une représentation lisible de cette taille (Go, Mo, Ko, o).
    """
    # Initialisation de la taille mémoire totale en octets
    taille_octets = 0  # 8 octets pour un élément float
    
    # Compteur pour le nombre de colonnes
    cpt = 0

    # Parcourt chaque colonne du dataframe
    for _, col_data in df.items():
        cpt += 1  # Incrémente le compteur de colonnes
        # Ajoute la taille mémoire pour les valeurs uniques de la colonne
        taille_octets += len(np.unique(col_data)) * 8

    # Conversion de la taille en une représentation lisible
    aff = decompose_taille(taille_octets)
    
    # Affichage des informations sur la taille
    print(f'{cpt} variable(s) : {taille_octets} octets {aff}')

    return taille_octets

def drawNaiveBayes(df, colonne):
    """
    Génère et dessine un graphe représentant la structure d'un classifieur Naïve Bayes 
    à partir d'un dataframe et du nom de la colonne qui représente la classe cible.

    Args:
        df (DataFrame) : Le dataframe contenant les données.
        colonne (str) : Le nom de la colonne représentant la classe cible.

    Returns:
        Une représentation graphique du modèle Naïve Bayes, 
        où la classe cible est liée à chaque autre colonne.
    """
    # Initialisation de la chaîne représentant les relations dans le graphe
    str = ""
    
    # Parcourt chaque colonne du dataframe
    for nom, _ in df.items():
        # Ignore la colonne cible 
        if nom == colonne:
            continue
        # Crée une relation de la classe cible vers la colonne courante
        curr = colonne + "->" + nom
        # Ajoute la relation au graphe
        if str == "":
            str += curr  # Première relation
        else:
            str = str + ";" + curr  # Ajout des relations suivantes
    
    # Retourne et dessine le graphe en utilisant la méthode u.drawGraph
    return u.drawGraph(str)

def nbParamsNaiveBayes(df, target, list_attr=False):
    """
    Calcule la taille mémoire nécessaire pour stocker les paramètres uniques 
    d'un modèle Naïve Bayes en fonction de la variable cible et d'une liste 
    d'attributs donnée (ou de toutes les colonnes du dataframe si aucune liste n'est fournie).

    Args:
        df (DataFrame) : Le dataframe contenant les données.
        target (str) : Le nom de la colonne représentant la variable cible.
        list_attr (list or bool, optional) : Une liste des noms des attributs 
                                             à inclure dans le calcul. Si `False`, 
                                             toutes les colonnes du dataframe sont prises en compte. 
                                             Par défaut, `list_attr=False`.

    Returns:
        int : La taille mémoire en octets pour stocker les paramètres uniques.
    """
    # Initialisation de la taille mémoire totale
    taille_octets = 0  # 8 octets par élément float

    # Nombre de valeurs uniques pour la variable cible (target)
    taille_target = len(np.unique(df[target]))

    # Si aucune liste d'attributs n'est fournie, calcul sur toutes les colonnes
    if list_attr == False:
        longueur = 0  # Nombre d'attributs traités
        for nom_col, col_data in df.items():
            longueur += 1  # Incrémente le compteur de colonnes
            if nom_col == target:
                # Ajoute la contribution de la variable cible seule
                taille_octets += taille_target
            else:
                # Ajoute la contribution de la probabilité conditionnelle
                taille_octets += taille_target * len(np.unique(col_data))
    else:
        # Si une liste d'attributs est spécifiée
        longueur = len(list_attr)
        if list_attr == []:
            # Cas particulier : liste vide
            taille_octets = taille_target
        for attr in list_attr:
            if attr == target:
                # Ajoute la contribution de la variable cible seule
                taille_octets += taille_target
            else:
                # Ajoute la contribution de la probabilité conditionnelle
                taille_octets += taille_target * len(np.unique(df[attr]))

    # Convertir la taille mémoire en octets
    taille_octets *= 8

    # Conversion en format lisible (Go, Mo, Ko, o)
    aff = decompose_taille(taille_octets)
    
    # Affichage des informations sur la taille
    print(f'{longueur} variable(s) : {taille_octets} octets {aff}')
    
    # Retourne la taille mémoire totale
    return taille_octets

def isIndepFromTarget(df, attr, seuil):
    """
    Vérifie si l'attribut spécifié 'attr' est indépendant de la variable cible 'target' 
    au seuil de p-value donné en pourcentage, en utilisant le test du chi-carré.

    Args:
        df (DataFrame) : Le dataframe contenant les données.
        attr (str) : Le nom de l'attribut à tester pour son indépendance avec 'target'.
        seuil (float) : Le seuil de p-value (en pourcentage) au-dessus duquel 
                        l'indépendance est acceptée.

    Returns:
        bool : Retourne True si l'attribut 'attr' est indépendant de 'target' 
               avec une p-value supérieure au seuil, sinon False.
    """
    # Création d'un tableau de contingence croisant 'target' et 'attr'
    cont = pd.crosstab(df['target'], df[attr])
    
    # Application du test du chi-carré pour tester l'indépendance
    s, p, dof, exp = chi2_contingency(cont)
    
    # Retourne True si la p-value est supérieure au seuil (indépendance acceptée)
    return p > seuil


def mapClassifiers(dic,df):
    """
    À partir d'un dictionnaire de classificateurs et d'un dataframe, cette fonction 
    représente graphiquement les classificateurs dans l'espace (précision, rappel).
    
    Args:
        dic (dict) : Un dictionnaire où les clés sont les noms des classificateurs 
                     et les valeurs sont les instances de ces classificateurs.
        df (DataFrame) : Le dataframe contenant les données à tester avec les classificateurs.
        
    Returns:
        ax : Un objet `Axes` contenant le graphique de l'espace (précision, rappel) pour les classificateurs.
    """

    # Initialisation des listes pour stocker les précisions et rappels
    x = []  # Liste pour la précision
    y = []  # Liste pour le rappel

    # Parcours des classificateurs dans le dictionnaire
    for i in dic:
        # Calcul des statistiques de performance du classificateur sur le dataframe
        d = dic[i].statsOnDF(df)
        
        # Ajout de la précision et du rappel dans les listes respectives
        x.append(d["Précision"])
        y.append(d["Rappel"])

    fig, ax = plt.subplots()
    ax.scatter(x, y, c ='red', marker = 'x')

    for i in range(len(x)):
        ax.annotate(str(i+1),(x[i], y[i]))
    return ax

def MutualInformation(df, x, y):
    """
    Calcule l'information mutuelle I(x; y) entre deux attributs x et y dans le dataframe df.

    Args:
        df (DataFrame) : Le dataframe contenant les données.
        x (str) : Le nom de l'attribut x.
        y (str) : Le nom de l'attribut y.
    
    Returns:
        float : La valeur de l'information mutuelle I(x; y) entre les attributs x et y.
    """
    # Initialisation des dictionnaires pour les distributions de probabilité
    px = dict()     # p(x)
    py = dict()     # p(y)
    pxy = dict()    # p(x, y)

    # Taille du dataframe (nombre de lignes)
    taille = int(df.size / len(df.keys()))

    # Construction des tableaux de contingence (comptage des occurrences)
    for i in range(taille):
        d = u.getNthDict(df, i)  # Obtient le i-ème dictionnaire (ligne du dataframe)
        
        # Comptage des occurrences de x
        if d[x] not in px:
            px[d[x]] = 1
        else:
            px[d[x]] += 1
        
        # Comptage des occurrences de y
        if d[y] not in py:
            py[d[y]] = 1
        else:
            py[d[y]] += 1
        
        # Comptage des paires (x, y)
        if (d[x], d[y]) not in pxy:
            pxy[(d[x], d[y])] = 1
        else:
            pxy[(d[x], d[y])] += 1

    # Normalisation des tableaux de contingence pour obtenir des probabilités
    for i in px:
        px[i] /= taille
    for i in py:
        py[i] /= taille
    for i in pxy:
        pxy[i] /= taille

    # Calcul de l'information mutuelle selon la formule I(x; y) = Σ p(x, y) * log2(p(x, y) / (p(x) * p(y)))
    s = 0
    for i in px:
        for j in py:
            if (i, j) in pxy:
                s += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))

    return s

def ConditionalMutualInformation(df, x, y, z):
    """
    Calcule l'information mutuelle conditionnelle I(x; y | z) dans le dataframe df.

    Args:
        df (DataFrame) : Le dataframe contenant les données.
        x (str) : Le nom de l'attribut x.
        y (str) : Le nom de l'attribut y.
        z (str) : Le nom de l'attribut z (condition).
    
    Returns:
        float : La valeur de l'information mutuelle conditionnelle I(x; y | z).
    """
    # Initialisation des structures pour stocker les fréquences et les distributions
    px = set()      # Ensemble des valeurs uniques de x
    py = set()      # Ensemble des valeurs uniques de y
    pz = dict()     # Distribution de probabilité pour z
    pxyz = dict()   # Distribution conjointe de (x, y, z)
    
    # Taille du dataframe
    taille = int(df.size / len(df.keys()))

    # Parcours du dataframe pour remplir les distributions
    for i in range(taille):
        d = u.getNthDict(df, i)  # Obtient le i-ème dictionnaire représentant une ligne
        
        # Remplissage des ensembles px et py avec les valeurs uniques de x et y
        px.add(d[x])
        py.add(d[y])
        
        # Comptage des occurrences de z
        if d[z] not in pz:
            pz[d[z]] = 1
        else:
            pz[d[z]] += 1
        
        # Comptage des occurrences des triplets (x, y, z)
        if (d[x], d[y], d[z]) not in pxyz:
            pxyz[(d[x], d[y], d[z])] = 1
        else:
            pxyz[(d[x], d[y], d[z])] += 1
    
    # Normalisation des tableaux de contingence pour obtenir des probabilités
    for i in pz:
        pz[i] /= taille  # p(z)
    for i in pxyz:
        pxyz[i] /= taille  # p(x, y, z)
    
    # Calcul de l'information mutuelle conditionnelle selon la formule :
    # I(x; y | z) = Σ p(x, y, z) * log2( p(z) * p(x, y, z) / (p(x, z) * p(y, z)) )
    s = 0
    for i in px:  # Parcours des valeurs de x
        for j in py:  # Parcours des valeurs de y
            for k in pz:  # Parcours des valeurs de z
                if (i, j, k) in pxyz:  # Si le triplet (i, j, k) existe dans pxyz
                    # Calcul de p(x, z) et p(y, z)
                    pxz = 0  # p(x, z)
                    pyz = 0  # p(y, z)
                    for (a, b, c) in pxyz:  # Parcours de tous les triplets dans pxyz
                        if a == i and c == k:
                            pxz += pxyz[(a, b, c)]  # Additionne les p(x, z)
                        if b == j and c == k:
                            pyz += pxyz[(a, b, c)]  # Additionne les p(y, z)
                    
                    # Calcul de la somme pour l'information mutuelle conditionnelle
                    s += pxyz[(i, j, k)] * np.log2(pz[k] * pxyz[(i, j, k)] / (pxz * pyz))

    return s

def MeanForSymetricWeights(a):
    """
    Calcule la moyenne des poids pour une matrice `a`. Cette fonction calcule la somme de tous les éléments de la matrice 
    et divise ensuite par le nombre d'éléments, en excluant les éléments de la diagonale (dans le cas de matrices symétriques, 
    la diagonale est ignorée car elle représente des poids auto-associés).

    Args:
        a (ndarray) : Une matrice de poids (de dimensions n x m).
    
    Returns:
        float : La moyenne des poids de la matrice `a`, excluant la diagonale (si applicable).
    """
    s = 0  # Initialisation de la somme des poids
    (n, m) = np.shape(a)  # Obtention des dimensions de la matrice a (n : nombre de lignes, m : nombre de colonnes)
    
    # Parcours de chaque élément de la matrice
    for i in range(n):
        for j in range(m):
            s += a[i][j]  # Ajout de la valeur de l'élément à la somme
    
    # Calcul de la moyenne des poids en excluant la diagonale (si applicable)
    s /= (n * (m - 1))  # Diviser par le nombre d'éléments, en excluant les éléments diagonaux
    
    return s  # Retourner la moyenne des poids


def SimplifyConditionalMutualInformationMatrix(a):
    """
    Annule toutes les valeurs plus petites que la moyenne des poids dans une matrice `a` symétrique de diagonale nulle.
    Cette fonction est utilisée pour simplifier une matrice de similarité ou d'information mutuelle conditionnelle 
    en supprimant les poids faibles, en les remplaçant par zéro.

    Args:
        a (ndarray) : Une matrice symétrique de taille n x m, représentant les poids (avec des valeurs positives) et 
                      ayant une diagonale nulle.

    Returns:
        None : La fonction modifie directement la matrice `a` sans renvoyer de nouvelle valeur.
    """
    # Calcul de la moyenne des poids en excluant la diagonale (avec la fonction MeanForSymetricWeights)
    mean = MeanForSymetricWeights(a)
    
    # Obtention des dimensions de la matrice a
    (n, m) = np.shape(a)
    
    # Parcours de chaque élément de la matrice
    for i in range(n):
        for j in range(m):
            # Si l'élément est inférieur à la moyenne des poids, il est annulé (remplacé par zéro)
            if a[i][j] < mean:
                a[i][j] = 0
    
    # La fonction ne retourne rien car elle modifie directement la matrice `a`
    return None

def Kruskal(df, a):
    """
    Propose la liste des arcs (non orientés pour l'instant) à ajouter dans notre classifieur sous la forme d'une liste de triplets (attr1, attr2, poids).
    Utilise l'algorithme de Kruskal pour trouver un sous-ensemble d'arcs de poids maximal qui connecte tous les nœuds sans former de cycles.
    
    Args: 
        df (DataFrame) : DataFrame contenant les attributs à connecter.
        a (ndarray) : Matrice des poids entre les attributs, utilisée pour déterminer la force de la connexion entre chaque paire d'attributs.
    
    Returns: 
        list : Liste des arcs à ajouter sous la forme [(attr1, attr2, poids)].
    """
    A = []                          # Liste des arcs sélectionnés (triplets : attribut1, attribut2, poids)
    noeuds = df.keys()              # Liste des attributs (nœuds)
    union_find = []                 # Liste pour implémenter la structure d'union-find (pour détecter les cycles)
    
    # Initialisation de l'union-find : chaque attribut est un ensemble séparé
    for i in noeuds:
        union_find.append({i})
    
    # Construction de la matrice des arêtes triées par poids décroissant
    AS = []                         # Liste des arêtes sous la forme (attribut1, attribut2, poids)
    (n, m) = np.shape(a)            # Dimensions de la matrice des poids
    for i in range(n):
        for j in range(i, m):        # On ne parcourt que la moitié supérieure de la matrice pour éviter les doublons
            if a[i][j] > 1e-15:     # On ne garde que les arêtes avec un poids significatif
                AS.append((noeuds[i], noeuds[j], a[i][j]))
    
    # Tri des arêtes par poids décroissant
    AS.sort(key=lambda tup: tup[2], reverse=True)  # Trie les arêtes selon le poids (élément [2])
    
    # Parcours des arêtes triées et ajout des arêtes qui ne forment pas de cycle
    for (u, v, x) in AS:  
        fu = set()  # Ensemble représentant l'ensemble de l'attribut u
        fv = set()  # Ensemble représentant l'ensemble de l'attribut v
        
        # Recherche de l'ensemble contenant u dans l'union-find
        for eu in union_find:
            if u in eu:
                fu = eu
                break
        
        # Recherche de l'ensemble contenant v dans l'union-find
        for ev in union_find:
            if v in ev:
                fv = ev
                break
        
        # Si les ensembles contenant u et v sont différents, cela signifie qu'ils ne sont pas encore connectés
        if fu != fv:
            # Ajouter l'arête (u, v) à la liste des arcs sélectionnés
            A.append((u, v, x))
            
            # Union des deux ensembles (fusion des ensembles de u et v)
            union_find.append(fu.union(fv))
            
            # Retirer les anciens ensembles de l'union-find
            if fu in union_find:
                union_find.remove(fu)
            if fv in union_find:
                union_find.remove(fv)
    
    return A  # Retourner la liste des arcs sélectionnés


def ConnexSets(L):
    """
    Rend une liste d'ensembles d'attributs connectés.
    Cette fonction prend une liste de triplets (attribut1, attribut2, poids) et retourne une liste d'ensembles d'attributs qui sont connectés, 
    en s'appuyant sur les paires d'attributs présentes dans la liste d'entrées.
    
    Args: 
        L (list) : Liste de triplets sous la forme (attribut1, attribut2, poids), où chaque triplet représente une connexion entre 
                  deux attributs avec un poids associé.
    
    Returns: 
        list : Liste d'ensembles, où chaque ensemble contient des attributs connectés entre eux.
    """
    res = []  # Liste des ensembles d'attributs connectés
    
    # Parcours de chaque triplet dans la liste L
    for (i, j, k) in L:
        b = True  # Indicateur pour savoir si un nouvel ensemble doit être créé
        
        # On vérifie si l'un des attributs i ou j est déjà dans un ensemble existant
        for ens in res:
            # Si l'attribut i ou j est déjà dans un ensemble, on les ajoute tous les deux à cet ensemble
            if i in ens or j in ens:
                ens.add(i)  # Ajoute i à l'ensemble
                ens.add(j)  # Ajoute j à l'ensemble
                b = False   # On ne crée pas un nouvel ensemble
                break       # On passe à la prochaine paire de la liste
            
        # Si aucun ensemble existant ne contient i ou j, on crée un nouvel ensemble
        if b:
            res.append({i, j})  # Création d'un nouvel ensemble avec i et j comme éléments
    
    return res  # Retourne la liste des ensembles d'attributs connectés


def OrientConnexSets(df, L, attr):
    """
    Compare l'information mutuelle des deux attributs par rapport à attr pour l'orienter.
    Cette fonction vérifie qu'un attribut ne peut pas avoir plus d'un seul parent en plus de target.
    Si un attribut doit avoir plusieurs parents, il conserve celui de poids maximum par rapport à target
    et devient le parent des autres (on inverse l'orientation des arcs).

    Args: 
        df (DataFrame)  : DataFrame contenant les données.
        L (list)        : Liste de triplets (attribut_a, attribut_b, poids), représentant des connexions non orientées entre les attributs.
        attr (str)      : Attribut étudié (souvent appelé "target"), qui sert de référence pour l'orientation des arcs.

    Returns: 
        list : Liste de paires (attribut_a, attribut_b), représentant les arcs orientés entre les attributs après traitement.
    """
    
    res = []    # Liste pour stocker les arcs orientés
    s = set()   # Ensemble des attributs qui ont déjà un parent

    # Parcours de chaque connexion non orientée dans la liste L
    for (x, y, a) in L:
        if y in s:  # Si l'attribut y a déjà un parent
            if x in s:  # Si x a aussi déjà un parent, on ignore cette paire
                continue
            res.append((y, x))  # On inverse l'orientation de l'arc
            s.add(x)  # On ajoute x à l'ensemble des attributs avec un parent
            continue

        if x in s:  # Si l'attribut x a déjà un parent
            if y in s:  # Si y a aussi déjà un parent, on ignore cette paire
                continue
            res.append((x, y))  # On inverse l'orientation de l'arc
            s.add(y)  # On ajoute y à l'ensemble des attributs avec un parent
            continue

        # Comparaison des informations mutuelles de x et y par rapport à l'attribut attr
        if MutualInformation(df, x, attr) > MutualInformation(df, y, attr):
            res.append((x, y))  # On choisit l'arc (x, y)
            s.add(y)  # On ajoute y à l'ensemble des attributs avec un parent
        else:
            res.append((y, x))  # On choisit l'arc (y, x)
            s.add(x)  # On ajoute x à l'ensemble des attributs avec un parent

    return res  # Retourne la liste des arcs orientés


class APrioriClassifier(u.AbstractClassifier):
    def __init__(self,data):
        super().__init__()
        self.data = data
    
    def estimClass(self, dico):
        """
        Estime la classe d'un objet en fonction de la probabilité estimée dans le dictionnaire d'entrées.

        Args:
            dico (dict): Dictionnaire contenant les informations sur l'estimation, typiquement issu de la fonction `getPrior`.
                        Le dictionnaire doit inclure une clé 'estimation' représentant la probabilité estimée.
                        Exemple : {'estimation': 0.7}
        
        Returns:
            int: Retourne 0 si l'estimation est inférieure à 0.5.
                Retourne 1 si l'estimation est supérieure ou égale à 0.5.
        """
        # Récupérer l'estimation du dictionnaire d'entrée
        dico = getPrior(self.data)
        estimation = dico['estimation']

        # Vérification de l'estimation et retour de la classe correspondante
        if estimation < 0.5:
            return 0  # Classe estimée : 0
        return 1  # Classe estimée : 1
    
    def statsOnDF(self, df):
        """
        Calcule les statistiques de classification (Vrai Positifs, Vrai Négatifs, Faux Positifs, Faux Négatifs, Précision, Rappel) à partir du dataframe.

        Args:
            df (pandas.DataFrame): Le dataframe contenant les données. Il doit inclure une colonne 'target' qui contient les valeurs réelles de la classe et les autres colonnes contenant les attributs des individus. Chaque ligne représente un individu avec sa classe réelle et ses attributs.

        Returns:
            dict: Un dictionnaire contenant les statistiques de classification, y compris:
                - 'VP' (Vrai Positifs): Le nombre d'individus correctement classés comme positifs.
                - 'VN' (Vrai Négatifs): Le nombre d'individus correctement classés comme négatifs.
                - 'FP' (Faux Positifs): Le nombre d'individus incorrectement classés comme positifs.
                - 'FN' (Faux Négatifs): Le nombre d'individus incorrectement classés comme négatifs.
                - 'Précision': La précision du modèle, calculée comme VP / (VP + FP).
                - 'Rappel': Le rappel du modèle, calculé comme VP / (VP + FN).
        """
        vp = 0 # Vrai Positifs
        vn = 0 # Vrai Négatifs
        fp = 0 # Faux Positifs
        fn = 0 # Faux Négatifs

        # Itération sur chaque ligne du dataframe
        for t in df.itertuples(index=False):
            dico = t._asdict()
            classePrevue = self.estimClass(dico)  # Estimation de la classe
            classeReelle = dico['target']  # Classe réelle
            # Mise à jour des compteurs en fonction de la prédiction et de la réalité
            if classePrevue == 1:
                if classeReelle == 1:
                    vp += 1
                else:
                    fp += 1
            else:
                if classeReelle == 1:
                    fn += 1
                else:
                    vn += 1

        # Calcul de la précision et du rappel
        precision = vp / (vp + fp) if (vp + fp) > 0 else 0
        rappel = vp / (vp + fn) if (vp + fn) > 0 else 0

        # Retourner les statistiques sous forme de dictionnaire
        return {'VP': vp, 
                'VN': vn,
                'FP': fp,
                'FN': fn,
                'Précision': precision,
                'Rappel': rappel}

    
class ML2DClassifier(APrioriClassifier):
    def __init__(self, data,attr):
        self.attr = attr
        self.p2dl = P2D_l(data,attr)
        super().__init__(data)

    def estimClass(self, dico):
        """
        Estime la classe d'un individu en fonction de la valeur d'un attribut et de la probabilité conditionnelle P(target|attr).

        Args:
            dico (dict): Dictionnaire représentant un individu et ses attributs. L'attribut spécifique à prédire (comme la classe cible) est accessible via `dico[self.attr]`.

        Returns:
            int: La classe prédite, soit 0 soit 1, selon laquelle la probabilité P(target|attr) est la plus élevée pour l'attribut donné.
        """
        value = dico[self.attr]  # Récupère la valeur de l'attribut à partir du dictionnaire
        max = -1  # Variable pour suivre la probabilité maximale

        # Parcours des deux classes (0 et 1)
        for t in [0, 1]:
            # Compare les probabilités P(target=t | attr=value)
            if self.p2dl[t][value] > max:
                max = self.p2dl[t][value]  # Mise à jour de la probabilité maximale
                att_max = t  # Classe avec la probabilité maximale

        return att_max  # Retourne la classe prédite
    
class MAP2DClassifier(APrioriClassifier):
    def __init__(self, data,attr):
        self.attr = attr
        self.p2dp = P2D_p(data,attr)
        super().__init__(data)

    def estimClass(self, dico):
        """
        Estime la classe d'un individu en fonction de la valeur d'un attribut et des probabilités conditionnelles P(attr|target).

        Args:
            dico (dict): Dictionnaire représentant un individu et ses attributs. L'attribut à prédire est accessible via `dico[self.attr]`.

        Returns:
            int: La classe prédite, soit 0 soit 1, selon laquelle la probabilité conditionnelle P(attr|target) est la plus élevée pour l'attribut donné.
        """
        value = dico[self.attr]  # Récupère la valeur de l'attribut à partir du dictionnaire
        max = -1  # Variable pour suivre la probabilité maximale

        # Parcours des deux classes (0 et 1)
        for t in [0, 1]:
            # Compare les probabilités P(attr=value | target=t)
            if self.p2dp[value][t] > max:
                max = self.p2dp[value][t]  # Mise à jour de la probabilité maximale
                att_max = t  # Classe avec la probabilité maximale

        return att_max  # Retourne la classe prédite

    
class MLNaiveBayesClassifier(APrioriClassifier):    #Qui utilise le maximum de vraisemblance (ML)
    # Pour estimer la classe d'un individu en utilisant l'hypothèse du Naïve Bayes
    def __init__(self,df):
        self.attrs = [col for col in df.columns if col != 'target']
        self.dic = dict() 
        for k in df.keys():
            if k != "target" and k != df.index.name:
                self.dic[k] = P2D_l(df,k) #Pour chaque attribut, on construit un dictionnaire de probabilité P(attr = a| target = t)
        super().__init__(df)
    
    def estimProbas(self, attrs):
        """
        Estime les probabilités conditionnelles P(target=0|attrs) et P(target=1|attrs) en fonction des valeurs des attributs fournis.

        Args:
            attrs (dict): Dictionnaire représentant les attributs de l'individu, où les clés sont les noms des attributs et les valeurs sont les valeurs des attributs correspondants. 
                        L'attribut "target" ne doit pas être inclus dans ce dictionnaire.

        Returns:
            dict: Dictionnaire avec les probabilités pour chaque classe `0` et `1`, 
                sous la forme {0: P(target=0|attrs), 1: P(target=1|attrs)}. Si une valeur d'attribut n'est pas présente dans les probabilités conditionnelles, 
                la probabilité est mise à zéro.
        """
        res0 = 1  # Probabilité P(target=0|attrs)
        res1 = 1  # Probabilité P(target=1|attrs)

        # Calcul des probabilités conditionnelles pour chaque attribut
        for i in attrs.keys():
            if i != "target":  # Ignore the target attribute
                d = self.dic[i]  # Dictionnaire des probabilités pour l'attribut
                if attrs[i] in d[1]:  # Si la valeur de l'attribut est dans le dictionnaire pour target=1
                    res1 *= d[1][attrs[i]]
                else:
                    res1 = 0  # Si la valeur n'est pas trouvée, la probabilité devient 0
                    break

                if attrs[i] in d[0]:  # Si la valeur de l'attribut est dans le dictionnaire pour target=0
                    res0 *= d[0][attrs[i]]
                else:
                    res0 = 0  # Si la valeur n'est pas trouvée, la probabilité devient 0
                    break

        return {0: res0, 1: res1}  # Retourne les probabilités pour target=0 et target=1


    def estimClass(self, dico):
        """
        Estime la classe cible (`target`) à partir des probabilités conditionnelles des attributs.

        Args:
            dico (dict): Dictionnaire représentant les attributs de l'individu, où les clés sont les noms des attributs et les valeurs sont les valeurs des attributs correspondants. 
                        L'attribut "target" ne doit pas être inclus dans ce dictionnaire.

        Returns:
            int: La classe estimée (0 ou 1). Si la probabilité pour `target=0` est plus grande que celle pour `target=1`, ou si les probabilités sont égales, la classe 0 est retournée. Sinon, la classe 1 est retournée.
        """
        d = self.estimProbas(dico)  # Estimation des probabilités pour target=0 et target=1
        if d[0] > d[1] or np.abs(d[0] - d[1]) < 1e-15:
            return 0  # Retourne 0 si la probabilité de target=0 est plus grande ou si les probabilités sont égales
        return 1  # Retourne 1 si la probabilité de target=1 est plus grande


class MAPNaiveBayesClassifier(APrioriClassifier):
    def __init__(self,df):
        self.df = df
        self.dic = dict() 
        self.pt = getPrior(self.df)['estimation']   #p(target = 1)
        for k in df.keys():
            if k != "target":
                self.dic[k] = P2D_l(df,k)           #Pour chaque attribut, on construit un dictionnaire de probabilité P(attr = a| target = t)

    def estimProbas(self, attrs):
        """
        Estime les probabilités conditionnelles P(target=0|attrs) et P(target=1|attrs) en fonction des valeurs des attributs fournis.
        
        Args:
            attrs (dict): Dictionnaire représentant les attributs de l'individu, où les clés sont les noms des attributs et les valeurs sont les valeurs des attributs correspondants. 
                        L'attribut "target" ne doit pas être inclus dans ce dictionnaire.
        
        Returns:
            dict: Dictionnaire avec les probabilités pour chaque classe `0` et `1`, 
                sous la forme {0: P(target=0|attrs), 1: P(target=1|attrs)}. Si les probabilités pour une classe sont trop faibles, elles sont normalisées. 
                Si les probabilités sont nulles, le dictionnaire retourne {0: 0, 1: 0}.
        """
        res1 = self.pt      # P(target = 1)
        res0 = 1 - res1     # P(target = 0)

        # Calcul des probabilités conditionnelles pour chaque attribut
        for i in attrs:
            if i != "target" and i != self.df.index.name:  # Ignore le target et l'index
                d = self.dic[i]  # Dictionnaire des probabilités pour l'attribut
                if attrs[i] in d[1]:  # Si la valeur de l'attribut est dans le dictionnaire pour target=1
                    res1 *= d[1][attrs[i]]
                else:
                    res1 = 0  # Si la valeur n'est pas trouvée, la probabilité devient 0
                    break

                if attrs[i] in d[0]:  # Si la valeur de l'attribut est dans le dictionnaire pour target=0
                    res0 *= d[0][attrs[i]]
                else:
                    res0 = 0  # Si la valeur n'est pas trouvée, la probabilité devient 0
                    break

        # Normalisation des probabilités
        if res0 > 1e-15 or res1 > 1e-15:
            return {0: res0 / (res0 + res1), 1: res1 / (res0 + res1)}
        return {0: 0, 1: 0}  # Si les probabilités sont nulles


    def estimClass(self, dico):
        """
        Estime la classe cible (`target`) à partir des probabilités conditionnelles des attributs.

        Args:
            dico (dict): Dictionnaire représentant les attributs de l'individu, où les clés sont les noms des attributs et les valeurs sont les valeurs des attributs correspondants. 
                        L'attribut "target" ne doit pas être inclus dans ce dictionnaire.

        Returns:
            int: La classe estimée (0 ou 1). Si la probabilité pour `target=0` est plus grande que celle pour `target=1`, ou si les probabilités sont égales, la classe 0 est retournée. Sinon, la classe 1 est retournée.
        """
        d = self.estimProbas(dico)  # Estimation des probabilités pour target=0 et target=1
        if d[0] > d[1] or np.abs(d[0] - d[1]) < 1e-15:
            return 0  # Retourne 0 si la probabilité de target=0 est plus grande ou si les probabilités sont égales
        return 1  # Retourne 1 si la probabilité de target=1 est plus grande

    
class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    #Même chose que MLNaiveBayesClassifier en ne considérant pas les individus indépendants
    def __init__(self, df, seuil):
        MLNaiveBayesClassifier.__init__(self,df)
        self.seuil = seuil
        self.df = df
        self.indep = set()
        for i in df.keys():
            if isIndepFromTarget(df,i,seuil):
                self.indep.add(i)

    def estimProbas(self, attrs):
        """
        Estime les probabilités conditionnelles en utilisant un modèle de Naive Bayes en excluant les attributs indépendants.

        Args:
            attrs (dict): Dictionnaire représentant les attributs de l'individu, où les clés sont les noms des attributs et les valeurs sont les valeurs des attributs correspondants. 
                        L'attribut "target" ne doit pas être inclus dans ce dictionnaire.

        Returns:
            dict: Dictionnaire avec les probabilités pour chaque classe `0` et `1`, sous la forme {0: P(target=0|attrs), 1: P(target=1|attrs)}.
        """
        b = attrs.copy() 
        for k in attrs:
            if k in self.indep:  # Si l'attribut est indépendant, il est retiré
                b.pop(k)

        # Appelle la méthode estimProbas de MLNaiveBayesClassifier pour estimer les probabilités
        return MLNaiveBayesClassifier.estimProbas(self,b)

    def draw(self):
        """
        Génère un graphique de dépendances des attributs non indépendants par rapport à l'attribut cible.
        """

        res = ""
        for i in self.df.keys():
            if i != "target":
                if i not in self.indep:
                    res+="target->"+i+";"
        return u.drawGraph(res)
    
class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    #Même chose que MAPNaiveBayesClassifier en ne considérant pas les individus indépendants
    def __init__(self, df, seuil):
        MAPNaiveBayesClassifier.__init__(self,df)
        self.seuil = seuil
        self.df = df
        self.indep = set()
        for i in df.keys():
            if isIndepFromTarget(df,i,seuil):
                self.indep.add(i)

    def estimProbas(self, attrs):
        b = attrs.copy() 
        for k in attrs:
            if k in self.indep:  # Si l'attribut est indépendant, il est retiré
                b.pop(k)
        return MAPNaiveBayesClassifier.estimProbas(self,b)

    def draw(self):
        """
        Génère un graphique de dépendances des attributs non indépendants par rapport à l'attribut cible.
        """
        res = ""
        for i in self.df.keys():
            if i != "target":
                if i not in self.indep:
                    res+="target->"+i+";"
        return u.drawGraph(res)
    
class MAPTANClassifier(MAPNaiveBayesClassifier):
    def __init__(self, df):
        MAPNaiveBayesClassifier.__init__(self, df) #Renvoie pt qui vaut P(target = 1) et dic qui contient un p2d_l pour chaque attribut
        cmis=np.array([[0 if x==y else ConditionalMutualInformation(df,x,y,"target") 
                for x in train.keys() if x!="target"]
                for y in train.keys() if y!="target"])
        SimplifyConditionalMutualInformationMatrix(cmis)
        #Calcul des arcs
        self.liste_arcs = OrientConnexSets(train, Kruskal(df, cmis), 'target') 
    
        self.dic2 = dict()      # Dictionnaire qui pour chaque attribut contient un dictionnaire P(fils|(pere, target)) de forme {attr : (pere,target) : fils : proba}
        self.enfant = set()     # Ensemble d'attr qui ont un pere en plus de target
        for (pere,fils) in self.liste_arcs:
            self.enfant.add(fils)
            self.dic2[(pere,fils)] = self.P_fils_sachant_parent(df,pere,fils)

    def P_fils_sachant_parent(self, df, pere, fils):
        """
        Calcule dans le dataframe la probabilité P(fils = f | (pere = p, target = t)) 
        sous la forme d'un dictionnaire associant à chaque couple (p, t) un dictionnaire 
        associant à chaque valeur f la probabilité P(fils = f | (pere = p, target = t)).

        Args:
            df (DataFrame): Le dataframe contenant les données à analyser.
            pere (str): Le nom de l'attribut correspondant au père.
            fils (str): Le nom de l'attribut correspondant au fils.

        Returns:
            dict: Un dictionnaire sous la forme :
                {(pere = p, target = t): {fils = f: P(fils = f | (pere = p, target = t))}}
        """
        # Construction d'une table de contingence sous la forme de dictionnaire {(pere, target): {fils: nb_occurences}}
        dic = dict()
        for i in range(int(df.size / len(df.keys()))):
            d = u.getNthDict(df, i)
            if (d[pere], d['target']) not in dic:
                dic[(d[pere], d['target'])] = dict()
            d2 = dic[(d[pere], d['target'])]
            if d[fils] not in d2:
                d2[d[fils]] = 1
            else:
                d2[d[fils]] += 1

        # Normalisation du dictionnaire sous la forme {(pere, target): {fils: proba}}
        for i in dic:
            d = dic[i]
            count = sum(d.values())
            for j in d:
                d[j] /= count

        return dic

    def estimProbas(self, attrs):
        """
        Estime les probabilités conditionnelles pour une liste d'attributs donnée, 
        en prenant en compte les relations parent-enfant et les cibles (target).

        Args:
            attrs (dict): Dictionnaire des valeurs des attributs à évaluer.

        Returns:
            dict: Un dictionnaire des probabilités pour chaque valeur de la cible (target),
                de la forme {0: prob_target_0, 1: prob_target_1}.
        """
        res1 = self.pt  # P(target = 1)
        res0 = 1 - res1  # P(target = 0)

        # Produit P(target) * P(attr | target) pour ceux sans parent
        for i in attrs:
            if i != "target" and i not in self.enfant:
                d = self.dic[i]
                if attrs[i] in d[1]:
                    res1 *= d[1][attrs[i]]
                else:
                    res1 = 0
                    break
                if attrs[i] in d[0]:
                    res0 *= d[0][attrs[i]]
                else:
                    res0 = 0
                    break

        # Produit P(fils | (pere, target)) pour ceux ayant deux parents
        for (parent, fils) in self.liste_arcs:
            d = self.dic2[(parent, fils)]
            if (attrs[parent], 0) in d:
                if attrs[fils] in d[(attrs[parent], 0)]:
                    res0 *= d[(attrs[parent], 0)][attrs[fils]]
                else:
                    res0 = 0
            if (attrs[parent], 1) in d:
                if attrs[fils] in d[(attrs[parent], 1)]:
                    res1 *= d[(attrs[parent], 1)][attrs[fils]]
                else:
                    res1 = 0

            # Cas où le tuple est absent du dictionnaire
            if (attrs[parent], 0) not in d and (attrs[parent], 1) not in d:
                return {0: 0, 1: 0}

        if res0 > 0 or res1 > 0:
            return {0: res0 / (res0 + res1), 1: res1 / (res0 + res1)}
        return {0: res0, 1: res1}

    def draw(self):
        """
        Génère une représentation graphique des relations parent-enfant dans les données.

        Returns:
            str: Une chaîne de caractères contenant les relations sous forme de graphe.
        """
        res = ""
        for i in self.df.keys():
            if i != "target":
                res += f"target->{i};"
        for (a, b) in self.liste_arcs:
            res += f"{a}->{b};"
        return u.drawGraph(res)



#####
# Question 2.4 : comparaison
#####
# Je préfère le classifieur MAP2DClassifier , car il repose sur la probabilité a posteriori (P(target∣attr)), 
# ce qui permet de prendre en compte à la fois la vraisemblance des observations et la distribution globale des classes. 
# Par exemple, en apprentissage, il atteint une précision de 87,1 % et un rappel de 82,2 %, tandis qu’en validation,
# la précision est de 85,7 % et le rappel de 82,6 %. Ces valeurs sont meilleures que celles de l’APrioriClassifier.
# l'APrioriClassifier, je le considère trop basique et limité en pratique,
# car il se contente de prédire la classe majoritaire sans tenir compte des caractéristiques des données 
# et donc ses performances ne sont pas satisfaisantes (une précision de 69 % en validation).
# Le ML2DClassifier est intéressant, mais ses résultats sont légèrement moins bons en validation : 
# une précision de 88,9 %, mais un rappel un peu plus faible à 81,8 %.
# Le ML2DClassifier peut être biaisé si les classes sont déséquilibrées.
#####

#####
# Question 3.3.a : preuve
#####
# Si on sait que A est indépendant de C sachant B: P(C|A,B) = P(C|B)
#    P(A,B,C) = P(A) * P(B|A) * P(C|A,B) = P(A) * P(B|A) * P(C|B)
#####
    
#####
# Question 3.3.b : complexité en indépendance partielle
#####
    # Sans indépendance conditionnelle: Puisque chaque variable A, B et C peut prendre 5 valeurs,
        # il y a 5*5*5=125 combinaisons possibles de valeurs pour les trois variables.
        # Donc, la taille mémoire nécessaire pour stocker la distribution conjointe est : 125 * 8 = 1000 octets
    # Avec indépendance conditionnelle ou P(A,B,C) = P(A) * P(B|A) * P(C|B)
        # taille P(A) : Il y a 5 valeurs possibles pour A, la taille mémoire nécessaire est donc : % * 8 = 40 octets
        # taille P(B|A) : Il y a 5 valeurs pour A et 5 valeurs pour B (matrice 5*5), la taille mémoire nécessaire est donc : 5 * 5 * 8 = 200 octets
        # taille P(C|B) : Il y a 5 valeurs pour A et 5 valeurs pour B (matrice 5*5), la taille mémoire nécessaire est donc : 5 * 5 * 8 = 200 octets
        # Donc, la taille mémoire totale nécessaire avec l'indépendance conditionnelle est : 200 + 200 + 40 = 440 octets
#####
    
#####
# Question 4.1 : exemples
#####
    # Indépendantes:
        # utils.drawGraphHorizontal("A;B;C:D;E")
    # Sans aucune indépendance:
        # utils.drawGraphHorizontal("A->B;B->C;C->D;D->E;E->A")
#####
    
#####
# Question 4.2 : naïve Bayes
#####
    # P(attr1, attr2, ... | target) = P(attr1 | target) * P(attr2 | target) * ... * P(attrk | target)
    # P(target | attr1, attr2, ...) = P(attr1, attr2, ... | target) * P(target) / P(attr1, attr2, ...)
    #                               = P(target) * P(attr1|target) * P(attr2|target) * ... / P(attr1, attr2, ...)
#####
    
#####
# Question 6.1 : Evaluation des classifieurs
#####
    # Le point idéal dans un graphe Précision-Rappel se situe au coin supérieur droit : Précision = 1.0 & Rappel = 1.0 
    # Donc, les performances des modèles se distribuent dans des compromis entre précision et rappel.
    # Pour comparer les différents classifieurs dans cette représentation graphique, je propose:
        # poids entre Précision et Rappel selon les objectifs:
            # si éviter les faux négatifs est prioritaire, il faut privilégier des classifieurs avec un rappel élevé 
            # si éviter les faux positifs est prioritaire, il faut privilégier des classifieurs avec une précision élevée
        # distance au point idéal (1.0 ,1.0):
            # calculer la distance euclidienne entre chaque point et (1.0,1.0)
            # le classifieur avec la distance la plus courte est le meilleur dans ce compromis global
#####
    
#####
# Question 6.3 : Conclusion
#####
    # En observant les deux graphiques, je peux conclure:
        # Le Classifieur A Priori : test & train comportament similaire
            # il a un rappel de 1 (il prédit toujours la classe majoritaire) mais une précision très faible
            # peu performant car il ne prend pas en compte les données
        # Les Classifieur ML2D et MAP2D : précision & rappel modérés
            # la précision augmente et que le rappel diminue sur l'ensemble d'entraînement, cela indique que le modèle devient de plus en plus précis dans ses prédictions, mais qu'il identifie moins de vrais positifs 
        # Les Classifieurs 4, 5, 6 et 7 :
            # aucune des méthodes n’atteint le point idéal, mais ces classifieurs semblent les plus fiables et performants
            # les modèles 4 et 6 s'en approchent le plus sur la base train 
            # les rappels sont faibles sur la base test, alors c'est possible que les modèles sont trop complexes ou sous-ajustés sur les données de test
            # donc les faibles rappels pour les classifieurs 4, 5, 6, et 7 sur la base de test indiquent probablement une mauvaise généralisation
    
                

