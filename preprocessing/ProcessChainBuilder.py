from preprocessing.Processor import Processor


class ProcessChainBuilder(object):

    def __init__(self):
        self.first_processor = None
        self.next_processor = None

    def next(self, next_processor):
        if self.first_processor is None:
            self.first_processor = next_processor
            self.next_processor = next_processor
        else:
            self.next_processor.next(next_processor)
            self.next_processor = next_processor

        return self

    def build(self):
        self.next_processor.next(Processor())
        return self.first_processor

