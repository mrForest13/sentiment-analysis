from preprocessing.Processor import Processor
from nltk import PorterStemmer


class DataStemmingProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.ps = PorterStemmer()

    def process(self, data):
        data['text'] = self.stem(data)

        return self.next_processor.process(data)

    def stem(self, data):
        return data['text'].apply(lambda x: [self.ps.stem(item) for item in x])
