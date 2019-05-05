from preprocessing.Processor import Processor


class JoinTokens(Processor):

    def __init__(self):
        super().__init__()

    def process(self, data):
        data['text'] = self.join(data)

        return self.next_processor.process(data)

    @staticmethod
    def join(data):
        return data['text'].apply(lambda x: " ".join(x))
