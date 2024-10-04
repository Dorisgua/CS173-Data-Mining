# CS173 Homework２

Name:陈欣禾

Student ID:2021533093

*（Please finish this report in English.）*

---

## Text Segmentation Method

* Please write down the name of the **word segmentation library** you chose:

*(Please write your answer here)*
Jieba
* Please briefly introduce the **word segmentation algorithm principle** of this library:

*(Please write your answer here)*


[//]: # (对于在预先存在的词典中找不到的词（称为“未登录词”），采用了隐马尔可夫模型（HMM）来评估其作为有效词语的潜力。该模型利用汉字的内在特性来预测字符序列形成连贯词语的可能性。然后，利用维特比算法，一种动态规划技术，确定这些未登录词的最可能切分方式。)
Jieba utilizes a prefix tree-based mechanism to efficiently scan the sentence and construct a directed acyclic graph (DAG) representing all possible word combinations. It employs dynamic programming to search for the maximum probability path, thereby identifying the optimal segmentation based on word frequencies.

For words that cannot be found in the pre-existing dictionary (referred to as "out-of-vocabulary" or "OOV" words), a Hidden Markov Model  is utilized to assess their potential as valid words. This model leverages the inherent properties of Chinese characters to predict the likelihood of character sequences forming coherent words. Subsequently, the Viterbi algorithm, a dynamic programming technique, is employed to determine the most probable segmentation for these OOV words.
## Resut Analysis

Choose **one** of the following to answer:

* Do you think there is still room for improvement in your results? If so, please analyze in which aspects can be improved?

* If you are satisfied with your results, please look ahead: With the existing data and current work, what further analysis can be conducted?

*(Please write your answer here)*
I choose the second question.

Our current work has provided us with the key words of each poet, reflecting the preferences of the entire poet group in their choice of vocabulary. We can further conduct sentiment analysis on the poems to understand the emotional nuances expressed by the poets, such as sadness, joy, or loneliness. By integrating the poets' backgrounds and circumstances, we can gain insights into their inner worlds. Additionally, we can create visually appealing visualizations, such as placing word clouds of different poets together, to further cluster and discover which poets are more similar in terms of emotion and vocabulary usage.

[//]: # (我们现在的工作得到了每个诗人的重要词，体现了整个诗人群体用词的偏好性。我们还可以对诗歌进行情感分析，了解诗人在作品中所表达的情感色彩，例如悲伤、喜悦、孤独等，结合诗人的背景和处境了解诗人的内心世界。此外可以做更加漂亮的可视化，例如将不同诗人的词云图放在一起，进一步聚类发现有哪些诗人在情感和用词方面比较相似。)