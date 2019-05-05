from preprocessing.Processor import Processor


class PunctuationsProcessor(Processor):

    def __init__(self, next_processor):
        super().__init__(next_processor)

    def process(self, data):
        processed_data = self.remove_punctuations(data)
        super().process(processed_data)
        return self.next_processor.process(processed_data)

    @staticmethod
    def remove_punctuations(data):
        return data['text'].apply(lambda x: " ".join([item for item in x.split() if item.isalpha()]))
