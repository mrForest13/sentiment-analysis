from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np


def plot_horizontal_bar(data, labels, x_label):
    y_pos = np.arange(len(labels))
    data_counts = data["sentiment"].value_counts()
    plt.barh(y_pos, data_counts, align='center', alpha=1)
    plt.yticks(y_pos, labels)
    plt.xlabel(x_label)

    plt.show()


def plot_vertical_bar(data, keys, x_label):
    y_pos = np.arange(len(data))
    plt.bar(y_pos, data)
    plt.xticks(y_pos, keys, rotation=45)
    plt.ylabel(x_label)
    plt.xlabel('Model')
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
