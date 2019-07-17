import pandas

from loader.Loader import Loader


class ResultsDataLoader(Loader):

    def __init__(self, csv_path):
        super().__init__(['negative', 'positive'])
        self.csv_path = csv_path

    def load(self):
        csv_path = self.csv_path

        columns = ['arline', 'review', 'amazon', 'model']
        self.data = pandas.read_csv(csv_path, header=None, names=columns, skiprows=[0])
