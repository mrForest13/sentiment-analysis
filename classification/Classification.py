import time

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(),
    'K Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Ada Boost': AdaBoostClassifier(),
    'SVM': SVC(),
}


class Classification(object):

    def __init__(self, folds=10, jobs=-1, score='f1'):
        self.scores = ['accuracy', 'precision', 'recall', 'f1']
        self.fit_results = {}
        self.predict_results = {}
        self.score = score
        self.folds = folds
        self.jobs = jobs

    def fit(self, name, train, train_labels, parameters=None):
        if parameters is None:
            raise ValueError("Parameters cannot be empty!")

        kf = StratifiedShuffleSplit(n_splits=self.folds)
        clf = GridSearchCV(classifiers[name], parameters[name], cv=kf, scoring=self.scores, n_jobs=self.jobs,
                           refit=self.score, verbose=10)
        start_time = time.time()
        clf.fit(train, train_labels)
        self.fit_results[name] = clf

        return ClassificationResult(clf, time.time() - start_time)

    def fit_all(self, train, train_labels, parameters=None):
        if parameters is None:
            raise ValueError("Parameters cannot be empty!")

        for name in classifiers.keys():
            self.fit(name, train, train_labels)

    def predict(self, name, test):
        self.predict_results[name] = self.fit_results[name].predict(test)

    def predict_all(self, test):
        for name in classifiers.keys():
            self.predict(name, test)


class ClassificationResult(object):

    def __init__(self, model, execution_time):
        index = model.best_index_
        self.best_param = model.best_params_
        self.execution_time = execution_time
        self.precision = model.cv_results_['mean_test_precision'][index]
        self.recall = model.cv_results_['mean_test_recall'][index]
        self.f1 = model.cv_results_['mean_test_f1'][index]
        self.accuracy = model.cv_results_['mean_test_accuracy'][index]

