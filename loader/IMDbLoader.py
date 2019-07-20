import glob
import os.path as path

import pandas

from loader.Loader import Loader


class IMDbLoader(Loader):

    def __init__(self, data_directory, pos_directory, neg_directory):
        super().__init__(['negative', 'positive'])
        self.data_directory = data_directory
        self.pos_directory = pos_directory
        self.neg_directory = neg_directory

    def load(self):
        neg_data = self.load_data(self.neg_directory, 'negative')
        pos_data = self.load_data(self.pos_directory, 'positive')

        self.data = pandas.DataFrame(neg_data + pos_data, columns=['sentiment', 'text'])
        self.data['text_length'] = [len(text) for text in self.data['text']]

    def load_data(self, directory, label):
        join_path = path.join(self.data_directory, directory, '*.txt')

        rows = []
        for name in glob.glob(join_path):
            with open(name, encoding='utf-8') as f:
                rows.append([label, f.read()])

        return rows
