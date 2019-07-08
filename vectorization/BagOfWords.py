from sklearn.feature_extraction.text import CountVectorizer
from vectorization.Vectorizer import Vectorizer


class BagOfWords(Vectorizer):

    def __init__(self, n):
        self.model = CountVectorizer(ngram_range=(n, n), tokenizer=lambda x: x.split())

    def fit_transform(self, data):
        return self.model.fit_transform(data)

    def transform(self, data):
        return self.model.transform(data)
