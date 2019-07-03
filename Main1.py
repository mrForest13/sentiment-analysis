from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_selection import mutual_info_classif

import numpy as np

import pandas as pd

print(format(3.18967567514651e-05, 'f'))
print('{:.15f}'.format(3.18967567514651e-05))

d = {'text': ["product is not good", "product is good", "you suck", "you bad", "great work"], 'sentiment': [0, 1, 0, 0, 1]}

df = pd.DataFrame(data=d)

print(df.head())

text = df['text']

labels = df['sentiment']

cv = CountVectorizer(ngram_range=(1, 1))

bag_of_words_train = cv.fit_transform(text)

print(bag_of_words_train.toarray())

feature = cv.get_feature_names()
results = mutual_info_classif(bag_of_words_train, labels, discrete_features=True)

print(list(zip(feature, results)))