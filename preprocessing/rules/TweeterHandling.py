from preprocessing.Processor import Processor

import re


class TweeterHandlingProcessor(Processor):

    def __init__(self, next_processor):
        super().__init__(next_processor)
        self.regex = r'@[A-Za-z0-9_]+'

    def process(self, data):
        processed_data = self.remove_handling(data)
        super().process(processed_data)
        return self.next_processor.process(processed_data)

    def remove_handling(self, data):
        return data['text'].apply(lambda x: re.sub(self.regex, '', x))
