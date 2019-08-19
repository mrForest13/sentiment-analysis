from setuptools import setup

setup(
    name='Sentiment Analysis',
    version='1.0',
    description='Sentiment Analysis master thesis UJ',
    author='ML',
    author_email='mateusz.kamil.ligeza@gmail.com',
    packages=['classification', 'plot', 'loader', 'preprocessing', 'vectorization'],
    install_requires=['scikit-learn', 'numpy', 'pandas', 'lxml', 'wordcloud', 'matplotlib', 'seaborn', 'swifter',
                      'nltk', 'gensim'],
)
