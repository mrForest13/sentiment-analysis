from sklearn.feature_extraction.text import TfidfVectorizer
from vectorization.Vectorizer import Vectorizer


class TfIdfModel(Vectorizer):

    def __init__(self, n, min_frequent=0):
        super().__init__(min_frequent)
        self.model = TfidfVectorizer(ngram_range=(n, n), tokenizer=lambda x: x.split())

    def fit_transform(self, data):
        fit_data = self.model.fit_transform(data)
        filtered_voc = self.filter_not_frequent(fit_data)
        self.model.vocabulary = dict(filtered_voc)

        return self.model.fit_transform(data)

    def transform(self, data):
        return self.model.transform(data)
