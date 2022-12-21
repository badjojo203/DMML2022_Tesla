# Unil Kaggle competition
## Team Tesla

**Members**: 
- Kabasa Jonathan 
- Osorio Santos Joël

## Description of the project
- Nothing yet

## Approach
- Nothing yet

## Results
- Nothing yet



<h1>Detecting the Difficulty Level of French Texts</h1>
<p>Le projet consiste à prédire le niveau de difficulté d'un texte en français (A1, A2, B1, B2, C1, C2). Pour cela, nous disposons d'un jeu de données d'entraînement comprenant les colonnes "phrases" et "difficulté", ainsi que d'un jeu de données comprenant uniquement la colonne "phrases" pour lequel nous devons prédire la difficulté en utilisant différentes approches.</p>
<h2>Data sets</h2>
<ul>
  <li>Jeu de données d'entraînement : comprend les colonnes "phrases" et "difficulté"</li>
  <li>Jeu de données de test : comprend uniquement la colonne "phrases" pour laquelle nous devons prédire la difficulté</li>
</ul>
<h2>Approches testées</h2>
<ol>
  <li>Nettoyage des données
    <ol>
      <li>Suppression des mots vides</li>
      <li>Nettoyage des données</li>
    </ol>
  </li>
  <li>Vectorisation
    <ol>
      <li>Hachage</li>
      <li>TF-IDF</li>
      <li>CountVectorizer</li>
    </ol>
  </li>
  <li>Classificateurs
    <ol>
      <li>Régression linéaire</li>
      <li>Régression logistique</li>
      <li>SVC linéaire</li>
      <li>Multinomial</li>
      <li>Classificateur par vote</li>
    </ol>
  </li>
</ol>

<h3>Nettoyage des données</h3>
<p>il s'agit d'un ensemble de techniques permettant de préparer les données en vue de leur utilisation pour l'analyse et le traitement. Le nettoyage des données peut inclure des opérations telles que la suppression de mots vides et le nettoyage des données.</p>
<h3>Vectoriseurs</h3>
<p> c'est une étape de prétraitement des données consistant à transformer des données textuelles en vecteurs de features numériques. Cela peut être utile pour les algorithmes de machine learning qui ne peuvent pas traiter directement des données textuelles. Il existe plusieurs méthodes de vectorisation, comme le hachage, TF-IDF et CountVectorizer.</p>
<h4>Hachage</h4>
<p>c'est une méthode de vectorisation qui transforme un texte en un vecteur de features numériques en utilisant une fonction de hachage. Le résultat est un vecteur de dimension fixe, indépendamment de la longueur du texte d'origine. Cette méthode est rapide, mais elle peut entraîner une perte de qualité en raison de la collision de hachage possible.</p>
<h4>TFIDF</h4>
<p>c'est une méthode de vectorisation qui mesure l'importance d'un mot dans un document par rapport à l'ensemble des documents d'un corpus. Elle utilise deux mesures : la fréquence du terme (TF) et l'inverse de la fréquence du terme dans le corpus (IDF).</p>
<h4>Count Vectorizer</h4>
<p> c'est une méthode de vectorisation qui compte le nombre d'occurrences de chaque mot dans un document et crée un vecteur de features numériques en conséquence</p>
<h3>Classifiers</h3>
<p>ce sont des algorithmes de machine learning utilisés pour prédire la classe d'appartenance d'un échantillon en fonction de ses caractéristiques. Il existe plusieurs types de classificateurs, tels que la régression linéaire, la régression logistique, SVC linéaire, multinomial et classificateur par vote.</p>
<h4>Régression logistique</h4>
<p>c'est un algorithme de machine learning utilisé pour prédire une variable binaire à partir de variables explicatives. Il consiste à modéliser la probabilité de l'événement cible en utilisant une fonction logistique.</p>
<h4>SVC linéaire</h4>
<p> c'est un algorithme de machine learning utilisé pour la classification binaire ou multi-classe. Il consiste à trouver un hyperplan de séparation optimal entre les différentes classes de données en utilisant une fonction de coût.</p>
<h4>Multinominal</h4>
<p> c'est un algorithme de machine learning utilisé pour la classification multi-classe. Il utilise un modèle de probabilité multinomial pour prédire la classe d'appartenance d'un échantillon.</p>
<h4>Classificateur par vote</h4>
<p>c'est un algorithme de machine learning qui utilise l'agrégation des prédictions de plusieurs classificateurs indépendants pour prédire la classe d'appartenance d'un échantillon. Il peut être utilisé pour améliorer la précision de la prédiction en combinant les forces de différents classificateurs.</p>

<h3>Méthodes de recherche</h3>
<h4>Recherche de la meilleur combinaison nettoyeur/vectoriseur/classifier</h4>
<p>Nous avons utilisé plusieurs méthodes pour trouver la meilleure combinaison de [nettoyage des données, vectorisation, classificateurs], notamment un algorithme qui essaie toutes les combinaisons.</p>
<h4>Recherche des paramètres optimaux</h4>
<p>Nous avons utilisé <code>GridSearch</code> pour trouver les meilleurs paramètres pour certains classificateurs.</p>
<h5>Train test split</h5>
<p>Nous avons créé une fonction qui essaie les tailles de test et les <code>random_state</code> optimaux pour le <code>train_test_split</code> pour chaque méthode.</p>