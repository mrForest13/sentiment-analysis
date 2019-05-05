from preprocessing.Processor import Processor


class FilterAsciiProcessor(Processor):

    def __init__(self, next_processor):
        super().__init__(next_processor)

    def process(self, data):
        processed_data = self.filter_ascii(data)
        super().process(processed_data)
        return self.next_processor.process(processed_data)

    @staticmethod
    def filter_ascii(data):
        return data['text'].apply(lambda x: x.encode('ascii', errors='ignore').decode("utf-8"))
