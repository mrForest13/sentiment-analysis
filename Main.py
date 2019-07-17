from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(X.shape)

print(vectorizer.get_feature_names())
print(vectorizer.vocabulary_)

print(X.toarray().sum(axis=0))

voc = np.zeros(9)

for x in X:
    for i, v in zip(x.indices, x.data):
        voc[i] += v

print(voc)