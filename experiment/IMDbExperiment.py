from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

from preprocessing.ProcessChainBuilder import ProcessChainBuilder
from preprocessing.cleaning.NegationHandling import NegationHandlingProcessor
from preprocessing.cleaning.Punctuations import PunctuationsProcessor
from preprocessing.cleaning.StopWords import StopWordsProcessor
from preprocessing.cleaning.TweeterHandling import TweeterHandlingProcessor
from preprocessing.cleaning.FilterHtmlLink import FilterHtmlLinkProcessor
from preprocessing.cleaning.FilterSpaces import FilterSpacesProcessor
from preprocessing.cleaning.HtmlEncoding import HtmlEncodingProcessor
from preprocessing.cleaning.LowercaseAll import LowercaseAllProcessor
from preprocessing.cleaning.FilterAscii import FilterAsciiProcessor
from preprocessing.normalization.DataLemmatizer import DataLemmatizer
from preprocessing.normalization.TokenizerData import TokenizerData
from preprocessing.normalization.JoinTokens import JoinTokens
from classification.Classification import Classification, classifiers
from sklearn.model_selection import train_test_split
from loader.IMDbLoader import IMDbLoader
from plot.Ploter import *

data_loader = IMDbLoader("C:\\Users\\mateu\\Downloads\\aclImdb\\train", 'pos', 'neg')

data_loader.load()

data = data_loader.get_data()

plot_pie(data, data_loader.labels)
plot_horizontal_bar(data, data_loader.labels, 'Liczba tweetów')
plot_box(data)

data = ProcessChainBuilder() \
    .next(LowercaseAllProcessor()) \
    .next(FilterAsciiProcessor()) \
    .next(FilterHtmlLinkProcessor()) \
    .next(HtmlEncodingProcessor()) \
    .next(TweeterHandlingProcessor()) \
    .next(NegationHandlingProcessor()) \
    .next(PunctuationsProcessor()) \
    .next(StopWordsProcessor()) \
    .next(FilterSpacesProcessor()) \
    .build() \
    .process(data)

plot_word_cloud(data, 'negative')
plot_word_cloud(data, 'positive')


def split_data(frame, size=0.2):
    return train_test_split(frame['text'], frame['sentiment'], test_size=size, stratify=frame['sentiment'])


data = ProcessChainBuilder() \
    .next(TokenizerData()) \
    .next(DataLemmatizer()) \
    .next(JoinTokens()) \
    .build() \
    .process(data)

classification = Classification(folds=10)

train, test, train_labels, test_labels = split_data(data)

cv = CountVectorizer(ngram_range=(1, 1))

bag_of_words_train = cv.fit_transform(train)
bag_of_words_test = cv.transform(test)

classification.fit_all(bag_of_words_train, train_labels)
classification.predict_all(bag_of_words_test)

best_accuracy = []

for name, result in classification.predict_results.items():
    acc = accuracy_score(test_labels, result)
    best_accuracy.append(acc)
    print("Accuracy for {}: {:.4%}".format(name, acc))

plot_vertical_bar(best_accuracy, classifiers.keys(), 'Dokładność')
