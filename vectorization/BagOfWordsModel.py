from sklearn.feature_extraction.text import CountVectorizer
from vectorization.Vectorizer import Vectorizer


class BagOfWordsModel(Vectorizer):

    def __init__(self, n, min_frequent=0):
        super().__init__(min_frequent)
        self.n = n
        self.model = CountVectorizer(ngram_range=(n, n), tokenizer=lambda x: x.split())

    def clean(self):
        self.__init__(self.n, self.min_frequent)

    def model_name(self):
        return "Bag of Word {}-grams".format(self.n)

    def fit_transform(self, data):
        fit_data = self.model.fit_transform(data)
        filtered_voc = self.filter_not_frequent(fit_data)
        self.model.vocabulary = dict(filtered_voc)

        return self.model.fit_transform(data)

    def transform(self, data):
        return self.model.transform(data)
