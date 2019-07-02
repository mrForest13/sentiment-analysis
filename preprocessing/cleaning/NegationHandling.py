import swifter
from preprocessing.Processor import Processor
from preprocessing.negations import Negations


class NegationHandlingProcessor(Processor):

    def process(self, data):
        data['text'] = self.transform_negations(data)

        return self.next_processor.process(data)

    @staticmethod
    def transform_negations(data):
        return data['text'].swifter.apply(lambda x: Negations.replace_negation(x))
