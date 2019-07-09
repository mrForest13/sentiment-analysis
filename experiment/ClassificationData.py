from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from classification.Classification import Classification
from loader.PreprocessedDataLoader import PreprocessedDataLoader
from plot.Ploter import plot_pie, plot_box

from vectorization.BagOfWordsModel import BagOfWordsModel
from vectorization.Doc2VecModel import Doc2VecModel
from vectorization.TdIdfModel import TfIdfModel


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
    # "review": review_loader,
    # "amazon": amazon_loader
}

model = 'Naive Bayes'


def predict(vectorizer):
    for name, loader in all_data.items():
        print("Start processing for {} ...".format(name))
        loaded_data = load_data(loader)

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

        return result


# uni_gram_bow = predict(BagOfWordsModel(1))
# bi_gram_bow = predict(BagOfWordsModel(2))
# tri_gram_bow = predict(BagOfWordsModel(3))
#
# uni_gram_td_idf = predict(TfIdfModel(1))
# bi_gram_td_idf = predict(TfIdfModel(2))
# tri_gram_td_idf = predict(TfIdfModel(3))

doc_2_vec_dm = predict(Doc2VecModel(dm=1))
doc_2_vec_dbow = predict(Doc2VecModel(dm=0))
