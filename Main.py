from gensim.models.doc2vec import TaggedDocument
from nltk import word_tokenize
from pandas import DataFrame
import uuid
from gensim.models.doc2vec import Doc2Vec

data = ["I love machine learning Its awesome",

        "I love coding in python",

        "I love building chatbots",

        "they chat amagingly well"]

docs = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

model = Doc2Vec(dm=1, vector_size=5, sample=0, window=2, min_count=1, workers=4, epochs=20)

print(docs)

model.build_vocab(docs)

model.train(docs, total_examples=model.corpus_count, epochs=model.iter)

print(model.docvecs[0])

test_data = word_tokenize("they chat amagingly well")
v1 = model.infer_vector(test_data)

print(v1)
