from preprocessing.Processor import Processor
from nltk.corpus import stopwords


class StopWordsProcessor(Processor):

    def __init__(self, next_processor, language='english'):
        super().__init__(next_processor)
        self.stop_words = stopwords.words(language)

    def process(self, data):
        super().process(data)

        processed_data = self.remove_stop_words(data)
        return self.next_processor.process(processed_data)

    def remove_stop_words(self, data):
        return data['text'].apply(lambda x: " ".join([item for item in x.split() if item not in self.stop_words]))
