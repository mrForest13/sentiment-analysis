from loader.IMDbLoader import IMDbLoader
from loader.MultiDomainLoader import MultiDomainLoader
from preprocessing.ProcessChainBuilder import ProcessChainBuilder
from preprocessing.cleaning.NegationHandling import NegationHandlingProcessor
from preprocessing.cleaning.Punctuations import PunctuationsProcessor
from preprocessing.normalization.StopWords import StopWordsProcessor
from preprocessing.cleaning.TweeterHandling import TweeterHandlingProcessor
from preprocessing.cleaning.FilterHtmlLink import FilterHtmlLinkProcessor
from preprocessing.cleaning.FilterSpaces import FilterSpacesProcessor
from preprocessing.cleaning.HtmlEncoding import HtmlEncodingProcessor
from preprocessing.cleaning.LowercaseAll import LowercaseAllProcessor
from preprocessing.cleaning.FilterAscii import FilterAsciiProcessor
from preprocessing.normalization.DataLemmatization import DataLemmatizationProcessor
from preprocessing.normalization.TokenizeData import TokenizeDataProcessor
from preprocessing.normalization.JoinTokens import JoinTokensProcessor
from loader.AirlineLoader import ArlineLoader
from plot.Ploter import *


def load_data(data_loader, plot=False):
    data_loader.load()

    data = data_loader.get_data()

    if plot:
        plot_pie(data, data_loader.labels)
        plot_box(data)

    return data


def clean_process(data, plot=False):
    processed_data = ProcessChainBuilder() \
        .next(LowercaseAllProcessor()) \
        .next(FilterAsciiProcessor()) \
        .next(FilterHtmlLinkProcessor()) \
        .next(HtmlEncodingProcessor()) \
        .next(TweeterHandlingProcessor()) \
        .next(NegationHandlingProcessor()) \
        .next(PunctuationsProcessor()) \
        .next(FilterSpacesProcessor()) \
        .build() \
        .process(data)

    if plot:
        plot_word_cloud(processed_data, 'negative')
        plot_word_cloud(processed_data, 'positive')

    return processed_data


def steaming(data):
    processed_data = ProcessChainBuilder() \
        .next(TokenizeDataProcessor()) \
        .next(DataLemmatizationProcessor()) \
        .next(JoinTokensProcessor()) \
        .next(StopWordsProcessor()) \
        .build() \
        .process(data)

    return processed_data


arline_loader = ArlineLoader("C:\\Users\\mateu\\Downloads\\airline\\Tweets.csv")
review_loader = IMDbLoader("C:\\Users\\mateu\\Downloads\\aclImdb\\train", 'pos', 'neg')
amazon_loader = MultiDomainLoader("C:\\Users\\mateu\\Downloads\\sorted_data", "positive.review", "negative.review")

all_data = {
    "arline_loader": arline_loader,
    "review_loader": review_loader,
    "amazon_loader": amazon_loader
}

for name, loader in all_data.items():
    print("Start processing {} ...".format(name))
    loaded_data = load_data(loader)
    cleaned_data = clean_process(loaded_data)
    normalized_data = steaming(cleaned_data)

    normalized_data['text_length'] = [len(text) for text in normalized_data['text']]
    normalized_data = normalized_data[normalized_data['text_length'] > 0]

    normalized_data.to_csv('processed/{}.csv'.format(name), encoding='utf-8', columns=['sentiment', 'text'])
    print("Finish processing {}".format(name))
