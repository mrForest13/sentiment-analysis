import numpy

classification = {
    'Naive Bayes': {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 1000, 10000],
    },
    'Logistic Regression': {
        'C': numpy.logspace(-4, 4, 20),
        'max_iter': [2000, 5000, 10000]
    },
    'K Neighbors': {
        'n_neighbors': list(range(1, 17, 2)),
        'weights': ['uniform', 'distance']
    },
    'Decision Tree': {
        'max_depth': list(range(15, 30, 2)),
        'min_samples_split': numpy.linspace(0.1, 1.0, 10),
        'min_samples_leaf': numpy.linspace(0.1, 0.5, 5),
        'max_features': [None, 'sqrt', 'log2']
    },
    'Random Forest': {
        'n_estimators': [50, 150, 300],
        'max_depth': list(range(15, 30, 2)),
        'min_samples_split': numpy.linspace(0.1, 1.0, 10),
        'min_samples_leaf': numpy.linspace(0.1, 0.5, 5),
        'max_features': [None, 'sqrt', 'log2']
    },
    'Ada Boost': {
        'n_estimators': list(range(10, 151, 10)),
    },
    'SVM': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        "gamma": [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
}