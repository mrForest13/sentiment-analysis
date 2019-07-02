from preprocessing.Processor import Processor

import swifter


class LowercaseAllProcessor(Processor):

    def process(self, data):
        data['text'] = self.lowercase_all(data)

        return self.next_processor.process(data)

    @staticmethod
    def lowercase_all(data):
        return data['text'].swifter.apply(lambda x: x.lower())
