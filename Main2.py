# Word2vec model for embeddings  
from gensim.models import Word2Vec
# For extracting pre-trained vectors  
from gensim.models import KeyedVectors
# PCA for dimensionality reduction
from sklearn.decomposition import PCA
# For ploting the results 
from matplotlib import pyplot

sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
             ['this', 'is', 'the', 'second', 'sentence'],
             ['yet', 'another', 'sentence'],
             ['one', 'more', 'sentence'],
             ['and', 'the', 'final', 'sentence']]

model_1 = Word2Vec(size=300, min_count=1)
#Feeding Our coupus
model_1.build_vocab(sentences)
#Lenth of the courpus
total_examples = model_1.corpus_count
#traning our model
print(total_examples)