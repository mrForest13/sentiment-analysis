from preprocessing.Processor import Processor


class LowercaseAllProcessor(Processor):

    def __init__(self, next_processor):
        super().__init__(next_processor)

    def process(self, data):
        processed_data = self.lowercase_all(data)
        super().process(processed_data)
        return self.next_processor.process(processed_data)

    @staticmethod
    def lowercase_all(data):
        return data['text'].str.lower()
