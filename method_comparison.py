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

from W2V_train import data_pat, model_pat


# 将文本处理成向量，返回np.array数组
def corpus_to_vec(corpus, model):
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


# 加载数据返回X，Y数据集
def load_data(data_path, model):
    corpus = []
    labels = []
    Classes = {}
    with open(data_path, encoding='utf8') as f:
        for line in f:
            line_l = line.split(',')
            sentence = line_l[2] + line_l[3]
            sentence = jieba.lcut(sentence)
            corpus.append(sentence)
            labels.append(line_l[1])
            if line_l[1] not in Classes and line_l[1] != 'category':
                Classes[line_l[1]] = len(Classes)
    corpus = corpus_to_vec(corpus[1:], model)
    labels = [Classes[i] for i in labels[1:]]
    return np.array(corpus), np.array(labels)


# 简单查看数据数量信息
def check_data(X_train, X_test, y_train, y_test):
    print('X_train.shape:', X_train.shape,
          'X_test.shape:', X_test.shape,
          'y_train.shape:', y_train.shape,
          'y_test.shape:', y_test.shape)
    print('训练集的类别数量:\n', pd.value_counts(y_train))
    print('测试集的类别数量:\n', pd.value_counts(y_test))


def main():
    model = Word2Vec.load(model_pat)
    X, Y = load_data(data_pat, model)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    check_data(X_train, X_test, y_train, y_test)
    classifiers = [SVC(kernel='linear'),
                   DecisionTreeClassifier(),
                   RandomForestClassifier()]
    for classifier in classifiers:
        start = time.time()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(classifier.__class__)
        print(classification_report(y_test, y_pred))
        print('used time:', time.time() - start)


if __name__ == "__main__":
    main()
    # model = Word2Vec.load(model_pat)
    # X, Y = load_data(data_pat, model)