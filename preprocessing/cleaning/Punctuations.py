from preprocessing.Processor import Processor


class PunctuationsProcessor(Processor):

    def process(self, data):
        data['text'] = self.remove_punctuations(data)

        return self.next_processor.process(data)

    @staticmethod
    def remove_punctuations(data):
        return data['text'].apply(lambda x: " ".join([item for item in x.split() if item.isalpha()]))
