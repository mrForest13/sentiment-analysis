from loader.ResultsDataLoader import ResultsDataLoader
from plot.Ploter import *

results_loader = ResultsDataLoader('results/naive_bayes.csv')

results_loader.load()

plot_vertical_bar(results_loader.get_data())
