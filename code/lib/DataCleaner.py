import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

"""
see documentation.html for more information about this class
"""


class DataCleaner:
    def __init__(self):
        self.cleaner = {
            'lemmatize': self.lemmatize,
            'data_cleaner': self.data_cleaner,
            'empty': self.empty,
            'remove_stop_words': self.remove_stop_words
        }

    @staticmethod
    def remove_stop_words(sentence):
        # Tokeniser la phrase et enlever les stop words
        words = nltk.word_tokenize(sentence)
        filtered_words = [word for word in words if word not in stopwords.words('french')]

        # Rejoindre les mots filtrés en une phrase
        return ' '.join(filtered_words)

    # Fonction pour lemmatiser les mots d'une phrase
    @staticmethod
    def lemmatize(sentence):
        # Tokeniser la phrase et lemmatiser les mots
        words = nltk.word_tokenize(sentence)
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(word) for word in words]

        # Rejoindre les lemmas en une phrase
        return ' '.join(lemmas)

    @staticmethod
    def data_cleaner(sms):
        # Define stopwords
        stop_words = stopwords.words('french')

        # Define tokenizer and stemmer
        from nltk.tokenize import word_tokenize
        from nltk.stem import PorterStemmer

        # Remove digits
        sms = re.sub(r"\d+", "", sms)

        # Lowercase
        sms = sms.lower()

        # Remove punctuation
        sms = str(re.sub(r"[^\w\s\d]", "", sms))

        # Remove stop words
        sms = sms.split()
        sms = " ".join([word for word in sms if not word in stop_words])

        # Tokenize
        sms = word_tokenize(sms)

        # Stemming
        ps = PorterStemmer()
        sms = [ps.stem(word) for word in sms]

        return str(sms)

    @staticmethod
    def empty(sentence):
        return sentence

    def clean(self, datamanager, names):
        for name in names:
            cleaner = self.cleaner[name]
            datamanager.X_train = [cleaner(sentence) for sentence in datamanager.X_train]
            datamanager.X_test = [cleaner(sentence) for sentence in datamanager.X_test]

        return datamanager

    def getCleaners(self, names):
        return [self.cleaner[name] for name in names]

    @staticmethod
    def unClean(datamanager):
        datamanager.X_test = datamanager.initial_X_test
        datamanager.X_train = datamanager.initial_X_train
        return datamanager
