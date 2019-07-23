import numpy
import pandas

from classification.Classification import Classification
from loader.PreprocessedDataLoader import PreprocessedDataLoader
from plot.Ploter import plot_pie, plot_box
from vectorization.BagOfWordsModel import BagOfWordsModel
from vectorization.Doc2VecModel import Doc2VecModel
from vectorization.TdIdfModel import TfIdfModel

arline_loader = PreprocessedDataLoader('processed/arline.csv')
review_loader = PreprocessedDataLoader('processed/review.csv')
amazon_loader = PreprocessedDataLoader('processed/amazon.csv')

all_data = {
    "arline": arline_loader,
    "review": review_loader,
    "amazon": amazon_loader
}

parameters = {
    'Naive Bayes': {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 100, 1000],
    },
    'Logistic Regression': {
        'C': numpy.logspace(-4, 4, 20),
        'max_iter': [1000, 5000, 10000]
    },
    'K Neighbors': {
        'n_neighbors': list(range(1, 17, 2)),
        'weights': ['uniform', 'distance']
    },
    'Ada Boost': {
        'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 150, 200],
    },
    'Decision Tree': {
        'max_depth': list(range(1, 20, 2)),
        'min_samples_split': numpy.linspace(0.1, 1.0, 10),
        'min_samples_leaf': numpy.linspace(0.1, 0.5, 5)
    },
    'Random Forest': {
        'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 150, 200],
        'max_depth': list(range(1, 20, 2)),
        'min_samples_split': numpy.linspace(0.1, 1.0, 10),
        'min_samples_leaf': numpy.linspace(0.1, 0.5, 5)
    },
    'SVM': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        "gamma": [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear', 'poly']
    }
}

model = 'Random Forest'


def load_data(data_loader, plot=False):
    data_loader.load()

    data = data_loader.get_data()

    data['sentiment'] = data['sentiment'].apply(lambda x: 0 if x == 'negative' else 1)

    if plot:
        plot_pie(data, data_loader.labels)
        plot_box(data)

    return data


def predict(vectorizer):
    result_dict = {}

    for name, loader in all_data.items():
        print("Start processing for {} and {} ...".format(name, vectorizer.model_name()))
        loaded_data = load_data(loader)

        print("Data length: {}".format(len(loaded_data)))

        print("Class number:")
        print(loaded_data['sentiment'].value_counts())

        classification = Classification(folds=5, score='f1')

        train = loaded_data.text
        train_labels = loaded_data.sentiment

        train_data = vectorizer.fit_transform(train)

        print("Train size {}".format(train_data.shape))

        result = classification.fit(model, train_data, train_labels, parameters)

        result_dict['model'] = vectorizer.model_name()

        result_dict["{} precision".format(name)] = result.precision
        result_dict["{} recall".format(name)] = result.recall
        result_dict["{} f1".format(name)] = result.f1
        result_dict["{} accuracy".format(name)] = result.accuracy

        result_dict['execution_time'] = result.execution_time

        params = {'{} {}'.format(name, k): v for k, v in result.best_param.items()}

        result_dict.update(params)

        vectorizer.clean()

        print("Train time {} s".format(round(result.execution_time, 2)))
        print("Finish processing for {} and {} ... \n".format(name, vectorizer.model_name()))

    file_name = model.lower().replace(" ", "_")
    model_name = vectorizer.model_name().lower().replace(" ", "_").replace("-", "_")
    result_frame = pandas.DataFrame(data=result_dict, index=[0])
    result_frame.to_csv('results/{}_{}.csv'.format(file_name, model_name), encoding='utf-8')
    return result_frame


uni_gram_bow = predict(BagOfWordsModel(n=1))
bi_gram_bow = predict(BagOfWordsModel(n=2, min_frequent=1))
tri_gram_bow = predict(BagOfWordsModel(n=3, min_frequent=1))

uni_gram_td_idf = predict(TfIdfModel(n=1))
bi_gram_td_idf = predict(TfIdfModel(n=2, min_frequent=1))
tri_gram_td_idf = predict(TfIdfModel(n=3, min_frequent=1))

doc_2_vec_dm = predict(Doc2VecModel(dm=1, size=300))
doc_2_vec_dbow = predict(Doc2VecModel(dm=0, size=300))

# frames = [
#     uni_gram_bow,
#     bi_gram_bow,
#     tri_gram_bow,
#     uni_gram_td_idf,
#     bi_gram_td_idf,
#     tri_gram_td_idf,
#     doc_2_vec_dm,
#     doc_2_vec_dbow
# ]
#
# file_name = model.lower().replace(" ", "_")
# final_frame = pandas.concat(frames)
# final_frame.to_csv('results/{}.csv'.format(file_name), encoding='utf-8')
#
# print("Final execution time for all models {} s\n".format(final_frame['execution_time'].sum()))
