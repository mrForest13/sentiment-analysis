from classification.Classification import Classification
from loader.PreprocessedDataLoader import PreprocessedDataLoader
from plot.Ploter import plot_pie, plot_box

from vectorization.BagOfWordsModel import BagOfWordsModel
from vectorization.Doc2VecModel import Doc2VecModel
from vectorization.TdIdfModel import TfIdfModel

import pandas


def load_data(data_loader, plot=False):
    data_loader.load()

    data = data_loader.get_data()

    data['sentiment'] = data['sentiment'].apply(lambda x: 0 if x == 'negative' else 1)

    if plot:
        plot_pie(data, data_loader.labels)
        plot_box(data)

    return data


arline_loader = PreprocessedDataLoader('processed/arline.csv')
review_loader = PreprocessedDataLoader('processed/review.csv')
amazon_loader = PreprocessedDataLoader('processed/amazon.csv')

all_data = {
    "arline": arline_loader,
    "review": review_loader,
    "amazon": amazon_loader
}

model = 'Naive Bayes'


def predict(vectorizer):
    result_dict = {}

    for name, loader in all_data.items():
        print("Start processing for {} and {} ...".format(name, vectorizer.model_name()))
        loaded_data = load_data(loader)

        print("Data length: {}".format(len(loaded_data)))

        classification = Classification(folds=10, score='f1')

        train = loaded_data['text']
        train_labels = loaded_data['sentiment']

        train_data = vectorizer.fit_transform(train)

        result = classification.fit(model, train_data, train_labels)

        print("Finish processing for {} and {} ...".format(name, vectorizer.model_name()))
        print()

        result_dict['model'] = vectorizer.model_name()

        result_dict["{} precision".format(name)] = result.precision
        result_dict["{} recall".format(name)] = result.recall
        result_dict["{} f1".format(name)] = result.f1
        result_dict["{} accuracy".format(name)] = result.accuracy

        result_dict['execution_time'] = result.execution_time

        params = {'{} {}'.format(name, k): v for k, v in result.best_param.items()}

        result_dict.update(params)

        vectorizer.clean()

    return pandas.DataFrame(data=result_dict, index=[0])


uni_gram_bow = predict(BagOfWordsModel(n=1))
bi_gram_bow = predict(BagOfWordsModel(n=2, min_frequent=1))
tri_gram_bow = predict(BagOfWordsModel(n=3, min_frequent=1))

uni_gram_td_idf = predict(TfIdfModel(n=1))
bi_gram_td_idf = predict(TfIdfModel(n=2, min_frequent=1))
tri_gram_td_idf = predict(TfIdfModel(n=3, min_frequent=1))

doc_2_vec_dm = predict(Doc2VecModel(dm=1))
doc_2_vec_dbow = predict(Doc2VecModel(dm=0))

frames = [
    uni_gram_bow,
    bi_gram_bow,
    tri_gram_bow,
    uni_gram_td_idf,
    bi_gram_td_idf,
    tri_gram_td_idf,
    doc_2_vec_dm,
    doc_2_vec_dbow
]

file_name = model.lower().replace(" ", "_")
pandas.concat(frames).to_csv('results/{}.csv'.format(file_name), encoding='utf-8')
