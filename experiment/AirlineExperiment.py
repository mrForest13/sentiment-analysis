from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

from preprocessing.ProcessChainBuilder import ProcessChainBuilder
from preprocessing.cleaning.NegationHandling import NegationHandlingProcessor
from preprocessing.cleaning.Punctuations import PunctuationsProcessor
from preprocessing.cleaning.TweeterHandling import TweeterHandlingProcessor
from preprocessing.cleaning.FilterHtmlLink import FilterHtmlLinkProcessor
from preprocessing.cleaning.FilterSpaces import FilterSpacesProcessor
from preprocessing.cleaning.HtmlEncoding import HtmlEncodingProcessor
from preprocessing.cleaning.FilterAscii import FilterAsciiProcessor
from preprocessing.normalization.DataLemmatization import DataLemmatizationProcessor
from preprocessing.normalization.TokenizeData import TokenizeDataProcessor
from preprocessing.normalization.JoinTokens import JoinTokensProcessor
from classification.Classification import Classification
from sklearn.model_selection import train_test_split
from loader.AirlineLoader import ArlineLoader

data_loader = ArlineLoader("C:\\Users\\mateu\\Downloads\\/airline/Tweets.csv")

data_loader.load()

data = data_loader.get_data()


def change(x):
    if x == 'neutral' or x == 'positive':
        return 1
    else:
        return 0


data['sentiment'] = data['sentiment'].apply(lambda x: change(x))

# plot_pie(data, data_loader.labels)
# plot_box(data)

data = ProcessChainBuilder() \
    .next(FilterAsciiProcessor()) \
    .next(FilterHtmlLinkProcessor()) \
    .next(HtmlEncodingProcessor()) \
    .next(TweeterHandlingProcessor()) \
    .next(NegationHandlingProcessor()) \
    .next(PunctuationsProcessor()) \
    .next(FilterSpacesProcessor()) \
    .build() \
    .process(data)


# plot_word_cloud(data, 'negative')
# plot_word_cloud(data, 'positive')


def split_data(frame, size=0.2):
    return train_test_split(frame['text'], frame['sentiment'], test_size=size, stratify=frame['sentiment'])


data = ProcessChainBuilder() \
    .next(TokenizeDataProcessor()) \
    .next(DataLemmatizationProcessor()) \
    .next(JoinTokensProcessor()) \
    .build() \
    .process(data)

classification = Classification(folds=10)

train, test, train_labels, test_labels = split_data(data)

cv = CountVectorizer(ngram_range=(1, 1))

bag_of_words_train = cv.fit_transform(train)
bag_of_words_test = cv.transform(test)

classification.fit_all(bag_of_words_train, train_labels)
classification.predict_all(bag_of_words_test)

for name, result in classification.predict_results.items():
    report = classification_report(test_labels, result)
    print("Classification Report for {}".format(name))
    print(report)

# plot_vertical_bar(best_accuracy, classifiers.keys(), 'Dokładność')
