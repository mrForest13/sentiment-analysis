import swifter
from preprocessing.Processor import Processor
from nltk.corpus import stopwords


class StopWordsProcessor(Processor):

    def __init__(self, language='english'):
        super().__init__()
        self.stop_words = stopwords.words(language)

    def process(self, data):
        data['text'] = self.remove_stop_words(data)

        return self.next_processor.process(data)

    def remove_stop_words(self, data):
        return data['text'].swifter.apply(
            lambda x: " ".join([item for item in x.split() if item not in self.stop_words]))
