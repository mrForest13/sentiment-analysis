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
from loader.AirlineLoader import ArlineLoader
from nltk.util import ngrams
from plot.Ploter import *
from preprocessing.normalization.DataLemmatization import DataLemmatizer
from preprocessing.normalization.TokenizeData import TokenizeData
import collections

from sklearn.feature_selection import mutual_info_classif

X = np.array([[0, 0, 0],
                [1, 1, 0],
                [2, 0, 1],
                [2, 0, 1],
                [2, 0, 1]])
y = np.array([0, 1, 2, 2, 1])

print(mutual_info_classif(X, y, discrete_features=True))

cos = ngrams('Mateusz ligeza cos tam'.split(), 1)
cos1 = ngrams('Mateusz ligeza cos tam gdziesz poszedl'.split(), 1)

print(collections.Counter(['ja', 'ja', 'ty']))

cos3 = set({k: v for k, v in dict(collections.Counter(['ja', 'ja', 'ty'])).items() if v > 1}.keys())

print(cos3)

data_loader = ArlineLoader("C:\\Users\\mateu\\Downloads\\airline\\Tweets.csv")

data_loader.load()

data = data_loader.get_data()

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

data = ProcessChainBuilder() \
    .next(TokenizeData()) \
    .next(DataLemmatizer()) \
    .build() \
    .process(data)

data['uni'] = data['text'].apply(lambda x: set(ngrams(x, 1)))
data['bi'] = data['text'].apply(lambda x: set(ngrams(x, 2)))
data['tri'] = data['text'].apply(lambda x: set(ngrams(x, 3)))

uni_gramy = data['uni'].to_numpy()

for i in range(1, len(uni_gramy)):
    uni_gramy[i] = uni_gramy[i - 1].union(uni_gramy[i])

for i in range(len(uni_gramy)):
    uni_gramy[i] = len(uni_gramy[i])

data['bi2'] = data['text'].apply(lambda x: list(ngrams(x, 2)))

sum = []

for i in data['bi2']:
    sum = sum + i

filtered = set({k: v for k, v in dict(collections.Counter(sum)).items() if v == 1}.keys())

bi_gramy = data['bi'].to_numpy()

for i in range(1, len(bi_gramy)):
    bi_gramy[i] = bi_gramy[i - 1].union(bi_gramy[i]).difference(filtered)

for i in range(len(bi_gramy)):
    bi_gramy[i] = len(bi_gramy[i])

data['tri2'] = data['text'].apply(lambda x: list(ngrams(x, 3)))

sum = []

for i in data['tri2']:
    sum = sum + i

filtered = set({k: v for k, v in dict(collections.Counter(sum)).items() if v == 1}.keys())

tri_gramy = data['tri'].to_numpy()

for i in range(1, len(tri_gramy)):
    tri_gramy[i] = tri_gramy[i - 1].union(tri_gramy[i]).difference(filtered)

for i in range(len(tri_gramy)):
    tri_gramy[i] = len(tri_gramy[i])

plt.title("Some Squiggles")
plt.plot(uni_gramy, label='uni-gram')
plt.plot(bi_gramy, label='bi-gram')
plt.plot(tri_gramy, label='tri-gram')
plt.legend(loc='upper left', frameon=True)
plt.ylabel('Liczba tweetów', fontsize=12)
plt.xlabel('Liczba unikatowych tokenów', fontsize=12)
#
plt.show()
