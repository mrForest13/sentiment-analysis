import abc


class Loader(object):

    def __init__(self, labels):
        self.col_names = ['sentiment', 'text']
        self.labels = labels
        self.data = None

    @abc.abstractmethod
    def load(self):
        pass

    def get_data(self):
        return self.data
