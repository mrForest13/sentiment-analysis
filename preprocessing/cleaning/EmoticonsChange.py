from preprocessing.Processor import Processor
from preprocessing.emoticons import Emoticons

import swifter


class EmoticonsChangeProcessor(Processor):

    def process(self, data):
        data['text'] = self.change_emoticons(data)

        return self.next_processor.process(data)

    @staticmethod
    def change_emoticons(data):
        return data['text'].swifter.apply(lambda x: Emoticons.replace_emoticon(x))
