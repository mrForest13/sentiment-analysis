from preprocessing.Processor import Processor


class FilterSpacesProcessor(Processor):

    def process(self, data):
        data['text'] = self.remove_unnecessary_spaces(data)

        return self.next_processor.process(data)

    @staticmethod
    def remove_unnecessary_spaces(data):
        return data["text"].swifter.apply(lambda x: " ".join(x.split()))
