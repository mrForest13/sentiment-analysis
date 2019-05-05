from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np


def plot_bar(data, labels, x_label):
    y_pos = np.arange(len(labels))
    data_counts = data["sentiment"].value_counts()
    plt.barh(y_pos, data_counts, align='center', alpha=1)
    plt.yticks(y_pos, labels)
    plt.xlabel(x_label)

    plt.show()


def plot_word_cloud(data_set, label):
    text = ' '.join([text for text in data_set['text'][data_set['sentiment'] == label]])
    word_cloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, background_color='white')
    plt.figure(figsize=(10, 7))
    plt.imshow(word_cloud.generate(text), interpolation="bilinear")
    plt.axis('off')
    plt.show()


def plot_box(data, y_label):
    plt.subplots(figsize=(5, 5))
    plt.boxplot(data['length'], sym='')
    plt.ylabel(y_label)
    plt.show()
