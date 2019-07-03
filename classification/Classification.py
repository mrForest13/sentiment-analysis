from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

import time
import numpy as np

classifiers = {
    'Naive Bayes': MultinomialNB(),
    # 'Logistic Regression': LogisticRegression(),
    # 'K Neighbors': KNeighborsClassifier(),
    # 'Decision Tree': DecisionTreeClassifier(),
    # 'Random Forest': RandomForestClassifier(),
    # 'Ada Boost': AdaBoostClassifier(),
    # 'SVM': SVC(),
}

parameters = {
    'Naive Bayes': {
        'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    },
    'Logistic Regression': {
        'C': np.logspace(-3, 3, 7),
        'solver': ['lbfgs'],
        'multi_class': ['auto'],
        'max_iter': [10000]
    },
    'K Neighbors': {
        'n_neighbors': list(range(1, 6)),
        'weights': ['uniform', 'distance'],
    },
    'Decision Tree': {
        'min_samples_split': range(10, 500, 20), 'max_depth': range(1, 20, 2)
    },
    'Random Forest': {
        'min_samples_split': [3, 5, 10],
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 15, 25],
        'max_features': [3, 5, 10, 20]
    },
    'Ada Boost': {
        'n_estimators': [50, 100, 200, 300],
    },
    'SVM': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        "gamma": [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
}

scorers = [
    'accuracy', 'precision', 'recall', 'f1'
]


class Classification(object):

    def __init__(self, folds=10, jobs=-1, score='f1'):
        self.fit_results = {}
        self.predict_results = {}
        self.score = score
        self.folds = folds
        self.jobs = jobs

    def fit(self, name, train, train_labels):
        kf = KFold(n_splits=self.folds, shuffle=True)
        clf = GridSearchCV(classifiers[name], parameters[name], cv=kf, scoring=scorers, n_jobs=self.jobs,
                           refit=self.score)
        start_time = time.time()
        clf.fit(train, train_labels)
        self.fit_results[name] = clf
        print("Execution time for fit {}: {}s".format(name, time.time() - start_time))
        print("Best params: " + str(clf.best_params_))
        print("Best scores: " + str(clf.best_score_))
        print()

    def fit_all(self, train, train_labels):
        for name in classifiers.keys():
            self.fit(name, train, train_labels)

    def predict(self, name, test):
        self.predict_results[name] = self.fit_results[name].predict(test)

    def predict_all(self, test):
        for name in classifiers.keys():
            self.predict(name, test)
