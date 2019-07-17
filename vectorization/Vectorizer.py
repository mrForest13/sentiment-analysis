import abc
import numpy
from sklearn.feature_extraction.text import CountVectorizer


class Vectorizer(object):

    def __init__(self, min_frequent):
        self.min_frequent = min_frequent
        self.model = None

    @abc.abstractmethod
    def clean(self):
        return

    def name(self):
        return "Tf-Idf"

    @abc.abstractmethod
    def fit_transform(self, data):
        return

    @abc.abstractmethod
    def transform(self, data):
        return

    def filter_not_frequent(self, data, n):
        model = CountVectorizer(ngram_range=(n, n), tokenizer=lambda x: x.split())

        fit_data = model.fit_transform(data)

        feature_names = model.get_feature_names()

        voc_sum = self.sum_tokens(fit_data, len(feature_names))

        print('Vocabulary before filter {}'.format(len(voc_sum)))

        zip_voc = list(zip(feature_names, voc_sum))

        vocabulary = list([(name, value) for name, value in zip_voc if value > self.min_frequent])
        filtered_voc = list([(value[0], i) for i, value in enumerate(vocabulary)])

        print('Vocabulary after filter {}'.format(len(filtered_voc)))

        return dict(filtered_voc)

    @abc.abstractmethod
    def model_name(self):
        return

    @staticmethod
    def sum_tokens(data, n):
        voc = numpy.zeros(n)

        for x in data:
            for i, v in zip(x.indices, x.data):
                voc[i] += v

        return voc

