# Sentiment Analysis

## 1. Data sets
- [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
- [Large Movie Review Dataset](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
- [Multi-Domain Sentiment Dataset](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/)

## 1. Project structure
The project is based on several modules, each of which is responsible for one task related to the sentiment analysis.

### Module loader
The module is responsible for loading data sets

### Module preprocessing
The package contains methods for text preprocessing. For example:
- [Punctuations](https://github.com/mrForest13/sentiment-analysis/blob/master/preprocessing/cleaning/Punctuations.py)
- [Emoticons](https://github.com/mrForest13/sentiment-analysis/blob/master/preprocessing/cleaning/EmoticonsChange.py)
- [Negation](https://github.com/mrForest13/sentiment-analysis/blob/master/preprocessing/cleaning/NegationHandling.py)
- [Stemming](https://github.com/mrForest13/sentiment-analysis/blob/master/preprocessing/normalization/DataStemming.py)
- [Lemmatization](https://github.com/mrForest13/sentiment-analysis/blob/master/preprocessing/normalization/DataLemmatization.py)

### Module vectorization
The module contains tokenization methods
- [Bag of Word](https://github.com/mrForest13/sentiment-analysis/blob/master/vectorization/BagOfWordsModel.py)
- [Tf-Idf](https://github.com/mrForest13/sentiment-analysis/blob/master/vectorization/TdIdfModel.py)
- [Doc2Vec](https://github.com/mrForest13/sentiment-analysis/blob/master/vectorization/Doc2VecModel.py)

### Module classification
Helper for the classification task based on GridSearchCV.

### Module plot
This module defines several basic charts that help show the structure of the data.

### Module experiment
This is the heart of the application (Main). We can find classes responsible for loading data and performing text preprocessing [here](https://github.com/mrForest13/sentiment-analysis/blob/master/experiment/PreprocessingData.py). For data classification [here](https://github.com/mrForest13/sentiment-analysis/blob/master/experiment/ClassificationData.py)
