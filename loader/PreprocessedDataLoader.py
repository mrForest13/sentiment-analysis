import pandas

from loader.Loader import Loader


class PreprocessedDataLoader(Loader):

    def __init__(self, csv_path):
        super().__init__(['negative', 'positive'])
        self.csv_path = csv_path

    def load(self):
        csv_path = self.csv_path

        self.data = pandas.read_csv(csv_path, header=None, names=['sentiment', 'text'], skiprows=[0])
