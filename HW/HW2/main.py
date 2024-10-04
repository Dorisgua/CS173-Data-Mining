# # coding=utf-8

import json
import jieba
import zhconv
import math
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
import re
# 直方图函数
def draw_histogram(poet_name, tfidf_scores):
    # TF-IDF降序排列
    sorted_tfidf = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    # 找到前十个
    top_10_words = [pair[0] for pair in sorted_tfidf[:10]]
    top_10_values = [pair[1] for pair in sorted_tfidf[:10]]
    # print(top_10_words)

    plt.figure(figsize=(10, 6))
    plt.barh(top_10_words, top_10_values, color='skyblue')

    font_prop = FontProperties(fname='data/simfang.ttf', size=12)
    plt.xlabel(f'TF-IDF Value', fontproperties=font_prop)
    plt.ylabel('Words', fontproperties=font_prop)
    plt.yticks(fontproperties=font_prop)
    plt.title(f'{poet_name} top 10 TF-IDF values', fontproperties=font_prop)
    # 保存
    plt.tight_layout()
    file_path = f'./result/histogram/{poet_name}.png'
    plt.savefig(file_path)
    plt.close()


# 词云图函数
def draw_word_cloud(poet_name, tfidf_scores):
    # TF-IDF降序排列
    sorted_tfidf = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    # 找top 50个
    top_50_words = {pair[0]: pair[1] for pair in sorted_tfidf[:50]}

    # 建立词云图
    wc = WordCloud(font_path='data/simfang.ttf')
    wc.generate_from_frequencies(top_50_words)

    # 保存
    file_path = f'./result/wordCloud/{poet_name}.png'
    wc.to_file(file_path)
    # print(f'Word cloud for {poet_name} saved successfully at {file_path}')

if __name__ == '__main__':
    print('Welcome to CS173 Homework2, please follow the homework description in readme.md\nHappy coding!')
    os.makedirs('./result', exist_ok=True)
    os.makedirs('./result/wordCloud', exist_ok=True)
    os.makedirs('./result/histogram', exist_ok=True)
    os.makedirs('./result/tf_idf', exist_ok=True)

    # chosen poets
    name_list = ['李白', '杜甫', '王维', '白居易', '李商隐', '韩愈', '元稹', '刘禹锡', '孟浩然', '韦应物']  # simplified

    """
    Implement your code here
    """
    # 1.1 Data Loading
    with open('data/tang.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    corpus = {}
    selected_poems={}
    poem_num = 0
    authors_num=0
    # 遍历 JSON 数据中的每首诗
    for line in data:
        # 1.2 Text Processing
        # 将诗的段落列表连接成一个字符串
        poem_num += 1
        poem_text = ''.join(line['paragraphs'])

        # 将繁体中文转换为简体中文
        author_simplified = zhconv.convert(line['author'], 'zh-cn')
        poem_text_simplified = zhconv.convert(poem_text, 'zh-cn')

        # 去除标点符号
        poem_text_cleaned = re.sub(r'[^\w\s]', '', poem_text_simplified)

        # 1.3 Participle
        # 使用结巴分词对诗文本进行分词
        words = list(jieba.cut(poem_text_cleaned))

        # 检查诗的作者是否已经在语料库字典中
        if author_simplified in corpus.keys():
            # 如果作者已经存在，将新的分词列表添加到已有列表后面
            authors_num += 1
            corpus[author_simplified] += words
        else:
            # 如果作者不存在，将作者作为键，分词列表作为值添加到字典中
            corpus[author_simplified] = words

    # 2.2 TF-IDF Calculation
    # 计算每位诗人的每个词的 TF 值
    poet_tf = {}  # 用于存储每位诗人的 TF 值的字典
    for poet, words in corpus.items():  # 遍历每位诗人及其对应的词列表
        if poet in name_list:
            # if poet == "李白":
            #     print("李白")
            #     print(len(words))
            word_count = len(words)  # 计算该诗人词列表中词的总数
            word_freq = {}  # 用于存储词频的字典
            for word in words:  # 遍历词列表中的每个词
                word_freq[word] = word_freq.get(word, 0) + 1  # 统计词频
            tf_scores = {word: freq / word_count for word, freq in word_freq.items()}  # 计算每个词的 TF 值
            poet_tf[poet] = tf_scores  # 将该诗人的 TF 值存储到字典中
    # print(poet_tf["李白"])

    # 计算每个词在所有诗人的诗歌中的 IDF 值
    word_poet_count = {}  # 用于存储每个词在不同诗人的诗歌中出现的次数的字典
    for poetof10, wordsof10 in corpus.items():
        if poetof10 in name_list:
            for poet, words in corpus.items():  # 遍历每位诗人及其对应的词列表
                for word in set(words):  # 使用集合去重，遍历诗人的词列表中的每个词
                    word_poet_count[word] = word_poet_count.get(word, 0) + 1  # 统计每个词在不同诗人的诗歌中出现的次数

    # print(authors_num)
    poet_idf = {word: math.log(authors_num / (count+1)) for word, count in word_poet_count.items()}  # 计算每个词的 IDF 值
    # if poet == "李白":
        #print(tfidf_scores)
    # print(poet_idf)
    # authors_num(诗人总数)和poem_num(是诗的总数)
    # 计算每位诗人的每个词的 TF-IDF 值
    poet_tfidf = {}
    for poet, tf_scores in poet_tf.items():
        tfidf_scores = {word: tf * poet_idf[word] for word, tf in tf_scores.items()}
        # 对 TF-IDF 值进行降序排序
        sorted_tfidf_scores = dict(sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True))
        poet_tfidf[poet] = sorted_tfidf_scores

    # 保存每位诗人的 TF-IDF 值到 JSON 文件中
    for poet, tfidf_scores in poet_tfidf.items():
        file_path = f'./result/tf_idf/{poet}.json'
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(tfidf_scores, json_file, ensure_ascii=False)

    # 3.1 Plot TF-IDF Histogram
    for poet_name, tfidf_scores in poet_tfidf.items():
        draw_histogram(poet_name, tfidf_scores)

    # 3.2 Draw Word Cloud Diagram
    for poet_name, tfidf_scores in poet_tfidf.items():
        draw_word_cloud(poet_name, tfidf_scores)




