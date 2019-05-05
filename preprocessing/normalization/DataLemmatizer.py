from preprocessing.Processor import Processor
from nltk.corpus.reader import NOUN
from nltk import WordNetLemmatizer


class DataLemmatizer(Processor):

    def __init__(self, pos=NOUN):
        super().__init__()
        self.pos = pos
        self.le = WordNetLemmatizer()

    def process(self, data):
        data['text'] = self.lemmatize(data)

        return self.next_processor.process(data)

    def lemmatize(self, data):
        return data['text'].apply(lambda x: [self.le.lemmatize(item, self.pos) for item in x])
