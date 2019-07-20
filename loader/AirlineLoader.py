import pandas

from loader.Loader import Loader


class ArlineLoader(Loader):

    def __init__(self, csv_path):
        super().__init__(['negative', 'neutral', 'positive'])
        self.csv_path = csv_path

    def load(self):
        csv_path = self.csv_path

        self.data = pandas.read_csv(csv_path, header=None, usecols=[1, 10], names=['sentiment', 'text'], skiprows=[0])
        self.data['text_length'] = [len(text) for text in self.data['text']]
