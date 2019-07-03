from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from classification.Classification import Classification
from loader.PreprocessedDataLoader import PreprocessedDataLoader
from plot.Ploter import plot_pie, plot_box


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

for name, loader in all_data.items():
    print("Start processing {} ...".format(name))
    loaded_data = load_data(loader)

    classification = Classification(folds=10, score='accuracy')

    train, test, train_labels, test_labels = split_data(loaded_data)

    cv = CountVectorizer(ngram_range=(1, 1))

    bag_of_words_train = cv.fit_transform(train)
    bag_of_words_test = cv.transform(test)

    classification.fit_all(bag_of_words_train, train_labels)
    classification.predict_all(bag_of_words_test)

    for model, result in classification.predict_results.items():
        report = classification_report(test_labels, result)
        print("Classification Report for {}".format(model))
        print(report)

    print("Finish processing {}".format(name))
