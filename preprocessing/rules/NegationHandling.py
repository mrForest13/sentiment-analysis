from preprocessing.Processor import Processor
from preprocessing.negations import Negations


class NegationHandlingProcessor(Processor):

    def __init__(self, next_processor):
        super().__init__(next_processor)

    def process(self, data):
        processed_data = self.transform_negations(data)
        super().process(processed_data)
        return self.next_processor.process(processed_data)

    @staticmethod
    def transform_negations(data):
        return data['text'].apply(lambda x: Negations.replace_negation(x))
