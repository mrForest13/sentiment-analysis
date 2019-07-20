import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud


def plot_vertical_bar(data, y_label='Dokładność'):
    bar_width = 0.25

    arline = data['arline accuracy'].tolist()
    review = data['review accuracy'].tolist()
    amazon = data['amazon accuracy'].tolist()

    models = data['model'].tolist()

    r1 = np.arange(len(arline))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    colors = sns.color_palette("deep")

    plt.bar(r1, arline, color=colors[0], width=bar_width, edgecolor='white', label='linie lotnicze')
    plt.bar(r2, review, color=colors[1], width=bar_width, edgecolor='white', label='recenzje (imdb)')
    plt.bar(r3, amazon, color=colors[2], width=bar_width, edgecolor='white', label='recenzje (amazon)')

    plt.xticks([r + bar_width for r in range(len(arline))], models, fontsize=9, rotation=90)

    plt.ylabel(y_label, fontsize=10)
    plt.legend(ncol=3)
    plt.ylim(top=1.15)
    plt.show()


def plot_word_cloud(data_set, label):
    text = ' '.join([text for text in data_set['text'][data_set['sentiment'] == label]])
    word_cloud = WordCloud(width=800, height=500, max_font_size=110, random_state=21, background_color='white',
                           stopwords=set(''))
    plt.figure(figsize=(10, 7))
    plt.imshow(word_cloud.generate(text), interpolation="bilinear")
    plt.axis('off')
    plt.show()


def plot_box(data):
    plt.subplots(figsize=(5, 5))
    plt.boxplot(data['text_length'], sym='')
    plt.ylabel('Liczba znaków')
    plt.show()


def plot_pie(data, labels):
    colors = sns.color_palette("pastel")
    data_count = data['sentiment'].value_counts()
    plot = data_count.plot(kind="pie", labels=translate_labels(labels), explode=[0.02] * len(labels), colors=colors,
                           autopct='%.2f%%')
    plot.set_ylabel("opinia")
    plt.show()


def translate_labels(labels):
    translate = {'negative': 'negatywna', 'neutral': 'neutralna', 'positive': 'pozytywna'}
    return map(lambda x: translate.get(x), labels)
