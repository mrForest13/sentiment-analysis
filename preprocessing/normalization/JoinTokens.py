from preprocessing.Processor import Processor


class JoinTokensProcessor(Processor):

    def __init__(self):
        super().__init__()

    def process(self, data):
        data['text'] = self.join(data)

        return self.next_processor.process(data)

    @staticmethod
    def join(data):
        return data['text'].swifter.apply(lambda x: " ".join(x))
