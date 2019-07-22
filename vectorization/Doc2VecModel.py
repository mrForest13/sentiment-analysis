from vectorization.Vectorizer import Vectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import numpy as np
import multiprocessing


class Doc2VecModel(Vectorizer):

    def __init__(self, dm=0, size=100, min_frequent=0):
        super().__init__(min_frequent)
        cores = multiprocessing.cpu_count()
        self.model = Doc2Vec(dm=dm, vector_size=size, sample=0, window=5, min_count=1, workers=cores, epochs=15)

    def clean(self):
        self.__init__(self.model.dm, self.model.vector_size, self.min_frequent)

    def model_name(self):
        return "D2V {}".format('PV-DM' if self.model.dm == 1 else 'PV-DBOW')

    def fit_transform(self, data):
        tagged_data = self.tagged_data(data)
        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        result = np.array([self.model.docvecs[str(i)] for i in range(len(tagged_data))])
        return (result - np.min(result))/np.ptp(result)

    def transform(self, data):
        tagged_data = self.tagged_data(data)
        return np.array([self.model.infer_vector(tagged_data[i][0]) for i in range(len(tagged_data))])

    @staticmethod
    def tagged_data(data):
        return [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(data)]
