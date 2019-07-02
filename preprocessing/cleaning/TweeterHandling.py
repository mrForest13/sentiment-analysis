import swifter
from preprocessing.Processor import Processor

import re


class TweeterHandlingProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.regex = r'@[A-Za-z0-9_]+'

    def process(self, data):
        data['text'] = self.remove_handling(data)

        return self.next_processor.process(data)

    def remove_handling(self, data):
        return data['text'].swifter.apply(lambda x: re.sub(self.regex, '', x))
