import os

import lxml.etree as tree
import pandas

from loader.Loader import Loader


class MultiDomainLoader(Loader):

    def __init__(self, data_directory, pos_file, neg_file):
        super().__init__(['negative', 'positive'])
        self.parser = tree.XMLParser(recover=True)
        self.data_directory = data_directory
        self.pos_file = pos_file
        self.neg_file = neg_file

    def load(self):
        results = []
        for category in self.list_dir():
            neg_data = self.load_data(category, self.neg_file, 'negative')
            pos_data = self.load_data(category, self.pos_file, 'positive')
            results = results + neg_data + pos_data

        self.data = pandas.DataFrame(results, columns=['sentiment', 'text'])
        self.data['text_length'] = [len(text) for text in self.data['text']]

    def load_data(self, category, directory, label):
        join_path = os.path.join(self.data_directory, category, directory)

        rows = []
        with open(join_path, errors='ignore') as file:
            root = tree.fromstringlist(["<root>", file.read(), "</root>"], parser=self.parser)
            for node in root:
                review_node = node.find("review_text")
                if review_node is not None:
                    rows.append([label, review_node.text])
        return rows

    def list_dir(self):
        result = os.listdir(self.data_directory)
        result.remove('stopwords')
        result.remove('summary.txt')
        return result
