import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet

from preprocessing.Processor import Processor


class DataLemmatizationProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.le = WordNetLemmatizer()
        self.tag_dict = {"j": wordnet.ADJ,
                         "n": wordnet.NOUN,
                         "v": wordnet.VERB,
                         "r": wordnet.ADV}

    def process(self, data):
        data['text'] = self.lemmatize(data)

        return self.next_processor.process(data)

    def lemmatize(self, data):
        return data['text'].swifter.apply(lambda x: [self.le.lemmatize(item, self.get_tag(item)) for item in x])

    def get_tag(self, word):
        tag = nltk.pos_tag([word])[0][1][0].lower()

        return self.tag_dict.get(tag, wordnet.NOUN)
