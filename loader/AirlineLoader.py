import pandas

from loader.Loader import Loader


class ArlineLoader(Loader):

    def __init__(self, csv_path):
        super().__init__(['negative', 'neutral', 'positive'])
        self.csv_path = csv_path

    def load(self):
        csv_path = self.csv_path
        col_names = self.col_names
        self.data = pandas.read_csv(csv_path, header=None, usecols=[1, 10], names=col_names, skiprows=[0])
