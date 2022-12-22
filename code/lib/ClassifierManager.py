import numpy as np

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .DataVectorizer import DataVectorizer

from .DataCleaner import DataCleaner

"""
see documentation.html for more information about this class
"""
class ClassifierManager:

    def __init__(self, ):
        self.classifier = {
            "naive_bayes": MultinomialNB(),
            "logistic_regression": LogisticRegression(),
            "random_forest": RandomForestClassifier(),
            "decision_tree": DecisionTreeClassifier(),
            "svm": SVC(),
            "knn": KNeighborsClassifier(),
            "linear_svc": LinearSVC(),
            "voting": VotingClassifier(estimators=[], voting='hard'),
        }

    def setClassifier(self, name, classifier):
        self.classifier[name] = classifier

    def displayNames(self):
        print("classifiers: ", list(self.classifier.keys()))

    def initVotingClassifier(self, classifiers_names):
        classifiers = self.getClassifier(classifiers_names)
        estimators = []
        for classifier in classifiers:
            estimators.append((classifier.__class__.__name__, classifier))
        self.classifier["voting"] = VotingClassifier(estimators=estimators, voting='hard')

    def getClassifiers(self, names):
        return [self.classifier[name] for name in names]

    def getClassifier(self, name):
        return self.classifier[name]

    @staticmethod
    def evaluate(y_test, y_pred):
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_pred)}")
        print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_pred):.4f}")
        print(f"CLASSIFICATION REPORT:\n\tPrecision: {precision:.4f}\n\tRecall: {recall:.4f}\n\tF1_Score: {f1:.4f}")

    def fit_predict(self, name, datamanager, verbose, params=None):
        classifier = self.classifier[name]

        if params is not None:
            classifier.set_params(**params)

        classifier.fit(datamanager.X_train, datamanager.y_train)

        # make predictions
        y_pred = classifier.predict(datamanager.X_test)

        # evaluate model
        if verbose:
            self.evaluate(datamanager.y_test, y_pred)

        return classifier

    @staticmethod
    def predict(datamanager, classifier):
        return classifier.predict(datamanager.X_test)

    def fit_predict_all(self, datamanager, verbose):
        for name in self.classifier:
            self.fit_predict(name, datamanager, verbose)

    def searchBestCombinations(self, datamanager, vectorizers_names, classifiers_names, data_cleaners_names):
        # Initialiser une liste pour stocker les résultats de chaque combinaison de méthodes
        results = []
        vectorizers = DataVectorizer().getVectorizers(vectorizers_names)
        classifiers = self.getClassifiers(classifiers_names)
        cleaners = DataCleaner().getCleaners(data_cleaners_names)

        max_algo_iter = len(vectorizers) * len(classifiers) * len(cleaners)
        current_iter = 0

        # Pour chaque combinaison de méthodes de vectorisation, de classification et de nettoyage de données
        for vec in vectorizers:
            for clf in classifiers:
                for cleaner in cleaners:
                    current_iter += 1


                    # Appliquer la méthode de nettoyage de données aux données d'entraînement et de test
                    X_train_cleaned = [cleaner(sentence) for sentence in datamanager.X_train]
                    X_test_cleaned = [cleaner(sentence) for sentence in datamanager.X_test]

                    # Convertir les phrases en vecteurs en utilisant la méthode de vectorisation
                    X_train_vec = vec.fit_transform(X_train_cleaned)
                    X_test_vec = vec.transform(X_test_cleaned)

                    # Entraîner un modèle de classification sur les données d'entraînement
                    clf.fit(X_train_vec, datamanager.y_train)

                    # Prédire les étiquettes pour les données de test
                    y_pred = clf.predict(X_test_vec)

                    # Calculer la précision du modèle sur les données de test
                    accuracy = accuracy_score(datamanager.y_test, y_pred)

                    precision = precision_score(datamanager.y_test, y_pred, average="macro")

                    # Ajouter les résultats à la liste de résultats
                    results.append({
                        'vectorizer': vec.__class__.__name__,
                        'classifier': clf.__class__.__name__,
                        'cleaner': cleaner.__name__,
                        'accuracy': accuracy,
                        "precision": precision
                    })

        # Trier les résultats par précision décroissante
        results.sort(key=lambda x: x['accuracy'], reverse=True)

        # Afficher les meilleurs résultats
        print("Best results:")
        for result in results[:5]:
            print("Vectorizer:", result['vectorizer'])
            print("Classifier:", result['classifier'])
            print("Cleaner:", result['cleaner'])
            print("Accuracy:", result['accuracy'])
            print()

    def gridSearch(self, classifier_name, param_grid, datamanager):
        classifier = self.classifier[classifier_name]

        grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')

        grid_search.fit(datamanager.X_train, datamanager.y_train)

        print("best params: ", grid_search.best_params_)
        print("precision :", grid_search.best_score_)
        print("test precision :", grid_search.score(datamanager.X_test, datamanager.y_test))
