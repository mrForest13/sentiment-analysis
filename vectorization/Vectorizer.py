import abc


class Vectorizer(object):

    @abc.abstractmethod
    def fit_transform(self, data):
        return

    @abc.abstractmethod
    def transform(self, data):
        return
