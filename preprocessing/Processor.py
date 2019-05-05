import abc


class Processor(object):

    def __init__(self, next_processor):
        self.next_processor = next_processor

    @abc.abstractmethod
    def process(self, data):
        if self.next_processor is None:
            return data
