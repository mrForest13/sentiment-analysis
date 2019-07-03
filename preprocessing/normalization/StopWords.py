from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from preprocessing.Processor import Processor
from nltk.corpus import stopwords

import pandas
import swifter


class StopWordsProcessor(Processor):

    def __init__(self, language='english'):
        super().__init__()
        self.stop_words = stopwords.words(language)
        self.cv = CountVectorizer(ngram_range=(1, 1))

    def process(self, data):
        text = data['text']
        sentiment = data['sentiment']

        counters = self.cv.fit_transform(text)

        results = mutual_info_classif(counters, sentiment, discrete_features=True)

        data['text'] = self.remove_stop_words(data, results)

        return self.next_processor.process(data)

    def remove_stop_words(self, data, results):
        features = self.cv.get_feature_names()

        d = {'text': features, 'result': results}

        df = pandas.DataFrame(data=d)
        df['result'] = df['result'].swifter.apply(lambda x: '{:.15f}'.format(x))
        df = df.sort_values(by=['result'])

        df.to_csv('processed/{}.csv'.format('cos'), encoding='utf-8', columns=['text', 'result'])

        return data['text']
