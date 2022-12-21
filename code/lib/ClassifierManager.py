import numpy as np
from nltk import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from DataVectorizer import DataVectorizer

from DataCleaner import DataCleaner


class ClassifierManager:

    def __init__(self, ):
        self.classifier = {
            "naive_bayes": MultinomialNB(),
            "logistic_regression": LogisticRegression(),
            "random_forest": RandomForestClassifier(),
            "svm": SVC(),
            "knn": KNeighborsClassifier(),
            "linear_svc": LinearSVC(),
        }

    def setClassifier(self, name, classifier):
        self.classifier[name] = classifier

    def displayNames(self):
        print("classifiers: ", list(self.classifier.keys()))


    def getClassifier(self, names):
        return [self.classifier[name] for name in names]

    @staticmethod
    def evaluate(datamanager, y_pred):
        precision = precision_score(datamanager.y_test, y_pred, average="macro")
        recall = recall_score(datamanager.y_test, y_pred, average="macro")
        f1 = f1_score(datamanager.y_test, y_pred, average="macro")
        print(f"CONFUSION MATRIX:\n{confusion_matrix(datamanager.y_test, y_pred)}")
        print(f"ACCURACY SCORE:\n{accuracy_score(datamanager.y_test, y_pred):.4f}")
        print(f"CLASSIFICATION REPORT:\n\tPrecision: {precision:.4f}\n\tRecall: {recall:.4f}\n\tF1_Score: {f1:.4f}")

    def fit_predict(self, name, datamanager, verbose):
        classifier = self.classifier[name]

        classifier.fit(datamanager.X_train, datamanager.y_train)

        # make predictions
        y_pred = classifier.predict(datamanager.X_train)

        # evaluate model
        if verbose:
            self.evaluate(datamanager.y_test, y_pred)

        return classifier

    def fit_predict_all(self, datamanager, verbose):
        for name in self.classifier:
            self.fit_predict(name, datamanager, verbose)

    def searchBestCombinations(self, datamanager, vectorizers_names, classifiers_names, data_cleaners_names):
        # Initialiser une liste pour stocker les résultats de chaque combinaison de méthodes
        results = []
        vectorizers = DataVectorizer().getVectorizers(vectorizers_names)
        classifiers = self.getClassifier(classifiers_names)
        cleaners = DataCleaner().getCleaners(data_cleaners_names)

        max_algo_iter = len(vectorizers) * len(classifiers) * len(cleaners)
        current_iter = 0

        # Pour chaque combinaison de méthodes de vectorisation, de classification et de nettoyage de données
        for vec in vectorizers:
            for clf in classifiers:
                for cleaner in cleaners:
                    current_iter += 1
                    print("===")
                    print("iter number " + str(current_iter) + " sur " + str(max_algo_iter))
                    print('vectorizer: ' + vec.__class__.__name__ + "\n" +
                          'classifier: ' + clf.__class__.__name__ + "\n" +
                          'cleaner: ' + cleaner.__name__)
                    print("---")
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

        return results[:5]

    def gridSearch(self, classifier_name, param_grid, datamanager):
        classifier = self.classifier[classifier_name]

        grid_search = GridSearchCV(classifier, param_grid)

        grid_search.fit(datamanager.X_train, datamanager.y_train)

        print("best params: ", grid_search.best_params_)
        print("pécision :", grid_search.best_score_)
        print("pécision avec test :", grid_search.score(datamanager.X_test, datamanager.y_test))


