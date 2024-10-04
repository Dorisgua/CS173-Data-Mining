# CS173 Homework２

This assignment will analyze the poetry of 10 poets. It requires calculating the TF-IDF values of the poems and visualizing them through histograms and word clouds. In the end, we will evaluate your code, TF-IDF results, visualization outcomes, and report content.

## Part 0：Data Introduction

* Data Sources：[chinese-poetry: 最全中文诗歌古典文集数据库](https://github.com/chinese-poetry/chinese-poetry)
* Data Size: 57607 poems
* File：
  * Path：`/data/tang.json`
  * Format：json
  * The data contains the following fields：
    * author
    * paragraphs
    * tags
    * title
    * id
  * Sample：

```json
    {
        "author": "李白",
        "paragraphs": [
            "昨夜誰爲吳會吟，風生萬壑振空林。",
            "龍驚不敢水中臥，猨嘯時聞巖下音。",
            "我宿黃山碧溪月，聽之却罷松間琴。",
            "朝來果是滄洲逸，酤酒醍盤飯霜栗。",
            "半酣更發江海聲，客愁頓向杯中失。"
        ],
        "tags": [
            "黄山"
        ],
        "title": "夜泊黃山聞殷十四吳吟",
        "id": "dc8cb4d0-bdb1-45ab-84ba-eff238065e00"
    }
```

---

## Part 1：Data Processing

### 1.1 Data Loading

* Combine all poems belonging to a poet into one document. (Please note: Each document corresponds to all the poems of a poet, not each poem)
  * Reference: Use the following code to read data and merge poems

```python
import json

with open('data/tang.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

corpus = {}
for line in data:
    if line['author'] in corpus.keys():
        corpus[line['author']] += line['paragraphs']
    else:
        corpus[line['author']] = line['paragraphs']
```

### 1.2 Text Processing

* Concatenate lists of verses into strings.
* Traditional and simplified character conversion. (`zhconv` library)
* Remove punctuation.

### 1.3 Participle

* Segment each poet’s poems into words.
  * Word segmentation is the process of recombining continuous word sequences into semantically independent word sequences according to certain specifications. In English writing, spaces are used as natural delimiters between words, while words in Chinese sentences do not have a formal delimiter and need to be segmented using a word segmentation algorithm.
* For convenience, we recommend using the [`jieba` library](https://github.com/fxsjy/jieba) for Chinese word segmentation, using the documentation and installation methods.
  You can use the `jieba.cut()` function.
* Of course, you can also choose other word segmentation libraries. We will briefly examine your understanding of word segmentation algorithms in the report.


* 将每位诗人的诗歌分割成单词。
分词是将连续的词序列按照一定的规范重新组合成语义独立的词序列的过程。在英文写作中，空格是词与词之间的自然分隔符，而中文句子中的词没有正式的分隔符，需要使用分词算法进行分词。
为方便起见，我们建议使用 jieba 库进行中文分词，并使用相关文档和安装方法。您可以使用 jieba.cut() 函数。
当然，您也可以选择其他分词库。我们将在报告中简要考察您对分词算法的理解。
---

## Part 2：TF-IDF

**Note: Please complete the calculation of TF-IDF independently. If a third-party library is used for calculation, this assignment will be scored as 0 points.**

### 2.1 Requirement

* In Part1, we built a corpus containing all poets. But we only focus on the following 10 poets:

```python
# selected poets
name_list = ['李白', '杜甫', '王维', '白居易', '李商隐', '韩愈', '元稹', '刘禹锡', '孟浩然', '韦应物']
```

* In subsequent calculations, we only calculate and visualize the TF-IDF of the poems of these 10 poets.

### 2.2 TF-IDF Calculation

* The calculated TF-IDF values should be stored in a descending dictionary list.
  * i.e. `{word1: value1, word2: value2, ...} where value1 > value2`。
* Then, save the calculated TF-IDF results for each document in json format in the `./result/tf_idf/` folder, using each poet's name as the file name.
  * For example: part of the file `./result/tf_idf/骆宾王.json` is:

```json
{"蝉声": TF-IDF value, "白头": TF-IDF value}
```

---

## Part 3：Visualization

### 3.1 Plot TF-IDF Histogram

* Visualize the TF-IDF value calculated in Part 2. Draw a histogram in descending order for the 10 words with the highest TF-IDF values in each poet's poetry collection. The words are marked on the horizontal axis, and the TF-IDF values are marked on the vertical axis.
* It is necessary to generate and draw 10 histograms, save the histograms in the `./result/histogram/` folder in the form of pictures, and use each poet's name as the file name.
* It is recommended to use the `matplotlib` library for drawing. When drawing the x-axis, you need to set the Chinese font file parameters. The solution is as follows：

``` python
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname='data/simfang.ttf', size=10) # The simfang.ttf we provide is placed in the data folder
plt.xticks(..., fontproperties=font)
```

### 3.2 Draw Word Cloud Diagram

* Generate a word cloud image for each poet and save it in the `./result/wordCloud/` folder, using each poet's name as the file name.
* For each word cloud, please select the top 50 words with the highest TF-IDF values for display.
* It is recommended to use the `wordcloud` library. To render Chinese in word cloud, you need to set the Chinese font file parameters. The solution is as follows:

```python
from wordcloud import WordCloud

wc = WordCloud(..., font_path='data/simfang.ttf')
```

---

## Part 4：Submit and Rate

| **Task**                        | **Scoring Criteria**                                                 | **Percentage** |
| ------------------------------- | ------------------------------------------------------------ | ------------ |
| **Coding（TF-IDF Calculation + Visualization）** | For TF-IDF calculation, results with word ranking order within a certain range of error compared to the reference code will be accepted; for visualization images, correctness and reasonableness will be examined. | 80%          |
| **Writing（Report）**           | Completeness of answers and reasonableness of analysis.         | 20%          |

### **File Requirements**

* Complete`main.py`, which needs to generate the following results:
  * `result/histogram/`: 10 histograms of TF-IDF.
  * `result/worldCloud/`: 10 word cloud images.
  * `result/tf_idf/`: 10 TF-IDF result files.
* Complete `report.md`。
* Finally, rename the CS173_Hw2 folder to CS173_Hw2_YourName_YourID, compress the folder, and submit the compressed document on Blackboard before 23:59 on March 28th.

---

## **Notes**

* Please complete the TF-IDF algorithm independently.
* Please maintain academic integrity.
