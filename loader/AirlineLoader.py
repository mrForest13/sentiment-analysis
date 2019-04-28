import pandas

from loader.Loader import Loader


class ArlineLoader(Loader):

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.labels = {'positive': 4, 'negative': 0, 'neutral': 2}

    def load(self):
        csv_path = self.csv_path
        col_names = self.col_names
        self.data = pandas.read_csv(csv_path, header=None, usecols=[1, 10], names=col_names, skiprows=[0])
        self.data['sentiment'] = self.data['sentiment'].apply(lambda row: self.labels[row])
