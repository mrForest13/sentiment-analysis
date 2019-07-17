import abc


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

    def filter_not_frequent(self, data):
        feature_names = self.model.get_feature_names()
        voc_sum = sum(data.toarray())

        print('Vocabulary before filter {}'.format(len(voc_sum)))

        zip_voc = list(zip(feature_names, voc_sum))

        vocabulary = list([(name, value) for name, value in zip_voc if value > self.min_frequent])
        filtered_voc = list([(value[0], i) for i, value in enumerate(vocabulary)])

        print('Vocabulary after filter {}'.format(len(filtered_voc)))

        return dict(filtered_voc)

    @abc.abstractmethod
    def name(self):
        return
