from preprocessing.Processor import Processor


class FilterSpacesProcessor(Processor):

    def __init__(self, next_processor):
        super().__init__(next_processor)

    def process(self, data):
        processed_data = self.remove_unnecessary_spaces(data)
        super().process(processed_data)
        return self.next_processor.process(processed_data)

    @staticmethod
    def remove_unnecessary_spaces(data):
        return data["text"].apply(lambda x: " ".join(x.split()))
