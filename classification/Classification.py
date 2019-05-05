from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

import time
import numpy as np

classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(),
    'K Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Ada Boost': AdaBoostClassifier(),
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


class Classification(object):

    def __init__(self, repeats=3, jobs=4):
        self.fit_results = {}
        self.predict_results = {}
        self.repeats = repeats
        self.jobs = jobs

    def fit(self, name, train, train_labels, scoring='accuracy'):
        clf = GridSearchCV(classifiers[name], parameters[name], cv=self.repeats, scoring=scoring, n_jobs=self.jobs)
        start_time = time.time()
        clf.fit(train, train_labels)
        self.fit_results[name] = clf
        print("Execution time for fit {}: {}s".format(name, time.time() - start_time))
        print("Best params: " + str(clf.best_params_))
        print("Best scores: " + str(clf.best_score_))
        print()

    def fit_all(self, train, train_labels, scoring='accuracy'):
        for name in classifiers.keys():
            self.fit(name, train, train_labels, scoring)

    def predict(self, name, test):
        self.predict_results[name] = self.fit_results[name].predict(test)

    def predict_all(self, test):
        for name in classifiers.keys():
            self.predict(name, test)
