from vectorization.Vectorizer import Vectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import numpy as np
import multiprocessing


class Doc2VecModel(Vectorizer):

    def __init__(self, dm=0, size=300):
        cores = multiprocessing.cpu_count()
        self.epochs = range(100)
        self.model = Doc2Vec(dm=dm, vector_size=size, sample=0, window=5, min_count=1, workers=cores, alpha=0.025,
                             min_alpha=0.00025, negative=5)

    def fit_transform(self, data):
        tagged_data = self.tagged_data(data)
        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        return np.array([self.model.docvecs[str(i)] for i in range(len(tagged_data))])

    def transform(self, data):
        tagged_data = self.tagged_data(data)
        return np.array([self.model.infer_vector(tagged_data[i][0]) for i in range(len(tagged_data))])

    @staticmethod
    def tagged_data(data):
        return [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(data)]
