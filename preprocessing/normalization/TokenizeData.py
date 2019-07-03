from preprocessing.Processor import Processor


class TokenizeDataProcessor(Processor):

    def __init__(self):
        super().__init__()

    def process(self, data):
        data['text'] = self.tokenize(data)

        return self.next_processor.process(data)

    @staticmethod
    def tokenize(data):
        return data['text'].apply(lambda x: x.split())
