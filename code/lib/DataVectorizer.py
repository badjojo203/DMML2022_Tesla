from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer


class DataVectorizer:
    def __init__(self):
        self.vectorizer = {
            "hash": HashingVectorizer(),
            "count": CountVectorizer(),
            "tfidf": TfidfVectorizer()
        }

    def setVectorizer(self, name, vectorizer):
        self.vectorizer[name] = vectorizer

    def getVectorizer(self, name):
        return self.vectorizer[name]

    def fit(self, name, datamanager):
        datamanager.X_test = self.vectorizer[name].fit_transform(datamanager.X_test)
        datamanager.X_train = self.vectorizer[name].fit_transform(datamanager.X_train)

    def unvectorize(self, datamanager):
        datamanager.X_test = datamanager.initial_X_test
        datamanager.X_train = datamanager.initial_X_train

    def getVectorizers(self, names):
        return [self.vectorizer[name] for name in names]
