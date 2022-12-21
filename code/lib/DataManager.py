import math
from .DataVectorizer import DataVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class DataManager:

    # data is a pandas dataframe
    def __init__(self, data, test_column_name):
        self.test_column_name = test_column_name
        self.y = data[str(test_column_name)]
        self.X = data.drop(test_column_name, axis=1)
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.data = data  # property that holds the data
        self.initial_data = data  # property that holds the initial data
        self.initial_y_test = None
        self.initial_y_train = None
        self.initial_X_test = None
        self.initial_X_train = None

    def labelEncode(self, column_name):
        # Label encode the column
        labelencoder = LabelEncoder()
        #
        self.data[column_name] = labelencoder.fit_transform(self.data[column_name])

        return self.data

    def splitData(self, test_size=0.2, random_state=0):
        # Split the data into training and testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                test_size=test_size,
                                                                                random_state=random_state)

        self.initial_X_train = self.X_train
        self.initial_X_test = self.X_test
        self.initial_y_train = self.y_train
        self.initial_y_test = self.y_test

        return self.X_train, self.X_test, self.y_train, self.y_test

    # method is the classifier
    # vectorizer is the vectorizer
    # test_size_array is an array of test size values e.g. [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    def otpimize_test_size(self, method, vectorizer, test_size_array):
        results = []
        # depth = 1 means 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        # depth = 2 means 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09
        # depth = 3 means 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009

        for test_size in test_size_array:
            X_train, X_test, y_train, y_test = train_test_split(self.data.X, self.data.y, test_size=test_size,
                                                                random_state=0)

            X_train = vectorizer.fit_transform(X_train)
            X_test = vectorizer.transform(X_test)

            method.fit(X_train, y_train)

            y_pred = method.predict(X_test)

            score = accuracy_score(y_test, y_pred)

            results.append({
                "test_size": test_size,
                "score": score,
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        best_params = results[0]["test_size"]
        best_score = results[0]['score']

        return best_params, best_score

    # method is the classifier
    # vectorizer is the vectorizer
    # random_state_array is an array of random states values e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    def otpimize_randomstate(self, method, vectorizer, random_state_array):
        results = []

        for random_state in random_state_array:
            X_train, X_test, y_train, y_test = train_test_split(self.data.X, self.data.y, test_size=0.2,
                                                                random_state=random_state)

            X_train = vectorizer.fit_transform(X_train)
            X_test = vectorizer.transform(X_test)

            method.fit(X_train, y_train)

            y_pred = method.predict(X_test)

            score = accuracy_score(y_test, y_pred)

            results.append({
                "random_state": random_state,
                "score": score,
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        best_params = results[0]["random_state"]
        best_score = results[0]['score']

        return best_params, best_score

    def inferDataToCSV(self, unlabelled_data, classifier, vectorizer_name, col_to_vectorize, col_to_drop, file_name):

        vectorizer = DataVectorizer().getVectorizer(vectorizer_name)

        unlabelled_test_data = unlabelled_data
        unlabelled_test_data_tfidf = vectorizer.transform(unlabelled_test_data[col_to_vectorize])
        y_pred = classifier.predict(unlabelled_test_data_tfidf)

        unlabelled_test_data[self.test_column_name] = y_pred

        unlabelled_test_data[self.test_column_name] = LabelEncoder().inverse_transform(y_pred)

        unlabelled_test_data.drop(columns=[col_to_drop], inplace=True)

        unlabelled_test_data.to_csv("../data/" + file_name + ".csv", index=False)
