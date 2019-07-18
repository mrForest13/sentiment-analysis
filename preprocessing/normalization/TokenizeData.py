from preprocessing.Processor import Processor

import swifter


class TokenizeDataProcessor(Processor):

    def __init__(self):
        super().__init__()

    def process(self, data):
        data['text'] = self.tokenize(data)

        return self.next_processor.process(data)

    @staticmethod
    def tokenize(data):
        return data['text'].swifter.apply(lambda x: x.split())
