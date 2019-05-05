import abc


class Processor(object):

    def __init__(self):
        self.next_processor = None

    def next(self, next_processor):
        self.next_processor = next_processor

    @abc.abstractmethod
    def process(self, data):
        return data
