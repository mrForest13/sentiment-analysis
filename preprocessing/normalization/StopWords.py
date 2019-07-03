from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from preprocessing.Processor import Processor

import pandas
import swifter


class StopWordsProcessor(Processor):

    def __init__(self, list_name, percentage_lvl=0.10):
        super().__init__()
        self.list_name = list_name
        self.percentage_lvl = percentage_lvl
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

        self.save_stopwords(features, results)

        zipped_result = sorted(zip(features, results), key=lambda x: x[1])

        max_value = int(len(zipped_result) * self.percentage_lvl)

        stopwords = [x[0] for x in zipped_result if x[1] < zipped_result[max_value][1]]

        return data['text'].swifter.apply(lambda x: " ".join([item for item in x.split() if item not in stopwords]))

    def save_stopwords(self, features, results):
        df = pandas.DataFrame(data={'text': features, 'result': results})
        df['result'] = df['result'].swifter.apply(lambda x: '{:.15f}'.format(x))
        df = df.sort_values(by=['result'])

        df.to_csv('processed/{}_stopwords.csv'.format(self.list_name), encoding='utf-8', columns=['text', 'result'])
