import swifter
from preprocessing.Processor import Processor


class FilterAsciiProcessor(Processor):

    def process(self, data):
        data['text'] = self.filter_ascii(data)

        return self.next_processor.process(data)

    @staticmethod
    def filter_ascii(data):
        return data['text'].swifter.apply(lambda x: x.encode('ascii', errors='ignore').decode("utf-8"))
