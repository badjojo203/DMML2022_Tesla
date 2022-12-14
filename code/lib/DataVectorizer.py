from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer

"""
see documentation.html for more information about this class
"""
class DataVectorizer:
    def __init__(self):
        self.vectorizer = {
            "hash": HashingVectorizer(),
            "count": CountVectorizer(),
            "tfidf": TfidfVectorizer(),

        }

    def setVectorizer(self, name, vectorizer):
        self.vectorizer[name] = vectorizer

    def getVectorizer(self, name):
        return self.vectorizer[name]

    def fit(self, name, datamanager):

        # print("print fils de grosse pute")
        vectorizer = self.vectorizer[name]
        #
        # print("euh lol ?")
        # print(TfidfVectorizer().transform(datamanager.initial_X_test["sentence"]).shape)

        datamanager.X_train = vectorizer.fit_transform(datamanager.initial_X_train)
        datamanager.X_test = vectorizer.transform((datamanager.initial_X_test))


    def unvectorize(self, datamanager):
        datamanager.X_test = datamanager.initial_X_test
        datamanager.X_train = datamanager.initial_X_train

    def getVectorizers(self, names):
        return [self.vectorizer[name] for name in names]
