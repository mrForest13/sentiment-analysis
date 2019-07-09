from sklearn.feature_extraction.text import TfidfVectorizer
from vectorization.Vectorizer import Vectorizer


class TfIdfModel(Vectorizer):

    def __init__(self, n):
        self.model = TfidfVectorizer(ngram_range=(n, n), tokenizer=lambda x: x.split())

    def fit_transform(self, data):
        return self.model.fit_transform(data)

    def transform(self, data):
        return self.model.transform(data)
