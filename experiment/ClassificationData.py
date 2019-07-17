from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
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


def split_data(frame, size=0.2):
    return train_test_split(frame['text'], frame['sentiment'], test_size=size, stratify=frame['sentiment'])


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

        print(len(loaded_data))

        classification = Classification(folds=10, score='accuracy')

        train, test, train_labels, test_labels = split_data(loaded_data)

        train_data = vectorizer.fit_transform(train)
        test_data = vectorizer.transform(test)

        classification.fit(model, train_data, train_labels)
        classification.predict(model, test_data)

        result = classification.predict_results[model]
        report = classification_report(test_labels, result)

        print("Classification Report for {}".format(model))
        print(report)

        print("Finish processing for {}".format(name))
        print()

        result_dict['model'] = vectorizer.model_name()
        result_dict[name] = [round(accuracy_score(test_labels, result), 4)]

        vectorizer.clean()

    return pandas.DataFrame(data=result_dict)


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
columns = ['arline', 'review', 'amazon', 'model']
pandas.concat(frames).to_csv('results/{}.csv'.format(file_name), encoding='utf-8', columns=columns)
