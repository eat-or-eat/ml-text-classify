import time
import jieba
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

"""
该脚本用于模型训练和结果查看
流程：加载词向量与语料->训练决策树、随机森林、SVC模型->查看效果
"""
# 路径配置
data_path = './data/tianchi_data.csv'
model_path = './output/model.w2v'
result_path = './output/result.txt'


def corpus_to_vec(corpus, model):
    """
    通过词向量相加的方式获取句向量
    :param corpus: 语料数组
    :param model: 词向量模型
    :return: 句向量
    """
    vectors = []
    for words in corpus:
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                pass
        vectors.append(vector / len(words))
    return vectors


def load_data(data_path, model):
    """
    加载数据函数
    :param data_path: 数据相对路径
    :param model: 词向量模型
    :return: 返回input数组和label数组
    """
    corpus = []
    labels = []
    Classes = {}
    with open(data_path, encoding='utf8') as f:
        for line in f:
            line_l = line.split(',')
            sentence = line_l[2]  # 获取第一个问题
            sentence = jieba.lcut(sentence)
            corpus.append(sentence)  # 获取标签名字
            labels.append(line_l[1])
            if line_l[1] not in Classes and line_l[1] != 'category':  # 跳过表头
                Classes[line_l[1]] = len(Classes)
    corpus = corpus_to_vec(corpus[1:], model)
    labels = [Classes[i] for i in labels[1:]]
    return np.array(corpus), np.array(labels)


def check_data(X_train, X_test, y_train, y_test):
    """
    检查查看数据形状
    """
    print('X_train.shape:', X_train.shape,
          'X_test.shape:', X_test.shape,
          'y_train.shape:', y_train.shape,
          'y_test.shape:', y_test.shape)
    print('训练集的类别数量:\n', pd.value_counts(y_train))
    print('测试集的类别数量:\n', pd.value_counts(y_test))


def main():
    model = Word2Vec.load(model_path)
    X, Y = load_data(data_path, model)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)  # 以0.3的比例切分训练测试数据
    check_data(X_train, X_test, y_train, y_test)

    classifiers = [SVC(kernel='linear'),  # 定义三种模型对比看看效果,留下的都是效果比较好的
                   # SVC(kernel='rbf'),
                   # SVC(kernel='sigmoid'),

                   DecisionTreeClassifier(),
                   # DecisionTreeClassifier(max_depth=10),
                   # DecisionTreeClassifier(max_depth=20),

                   RandomForestClassifier(),
                   # RandomForestClassifier(n_estimators=10),
                   # RandomForestClassifier(n_estimators=15)
                   ]

    writer = open(result_path, 'w', encoding='utf8')
    for classifier in classifiers:
        start_time = time.time()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        model = classifier.__class__
        result = classification_report(y_test, y_pred)
        use_time = time.time() - start_time
        print(model)
        print('used time:', use_time)
        print(result)
        writer.write("模型：%s \n 耗时：%fs \n 结果： %s" % (model, use_time, result))
    writer.close()


if __name__ == "__main__":
    # 测试用例
    # model = Word2Vec.load(model_pat)
    # X, Y = load_data(data_pat, model)

    main()
