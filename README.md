# ml-text-classify
使用sklearn的机器学习算法对文本进行分类

进度

- [x] 基本baseline-数据处理，模型训练，模型预测 -21/12/8

- [ ] 详细数据集分析 -待完成，等复习完后面的回头再来弄
- [ ] 数据增强 -同上
- [ ] 模型调参优化 - 同上
- [ ] 导出训练结果 - 同上

# 一，使用项目

环境

```bash
gensim==4.1.2
jieba==0.42.1
numpy==1.18.5
pandas==1.0.5
scikit_learn==1.0.1
```

## 1.下载

`git clone git@github.com:eat-or-eat/ml-text-classify.git`

## 2.（选）如果改用自己的数据需要修改method_comparison.load_data()函数读取数据

## 3.运行

`python ./method_comparison.py`

打印示例:

```bash
<class 'sklearn.svm._classes.SVC'>
              precision    recall  f1-score   support
           0       0.60      0.33      0.43       117
           1       0.95      0.96      0.96       405
           2       0.84      0.83      0.84       316
           3       0.84      0.88      0.86       368
           4       0.69      0.63      0.66       249
           5       0.70      0.80      0.74       329
           6       0.46      0.65      0.53       186
           7       0.00      0.00      0.00        38
           8       0.88      0.69      0.77        96
           9       0.00      0.00      0.00         6
    accuracy                           0.77      2110
   macro avg       0.60      0.58      0.58      2110
weighted avg       0.76      0.77      0.76      2110
used time: 0.7988836765289307

<class 'sklearn.tree._classes.DecisionTreeClassifier'>
              precision    recall  f1-score   support
           0       0.40      0.47      0.43       117
           1       0.93      0.91      0.92       405
           2       0.80      0.74      0.77       316
           3       0.75      0.74      0.74       368
           4       0.56      0.61      0.59       249
           5       0.69      0.70      0.70       329
           6       0.44      0.40      0.42       186
           7       0.31      0.29      0.30        38
           8       0.36      0.42      0.38        96
           9       0.33      0.17      0.22         6
    accuracy                           0.68      2110
   macro avg       0.56      0.54      0.55      2110
weighted avg       0.69      0.68      0.68      2110
used time: 0.5595088005065918
<class 'sklearn.ensemble._forest.RandomForestClassifier'>
              precision    recall  f1-score   support
           0       0.69      0.50      0.58       117
           1       0.95      0.96      0.95       405
           2       0.87      0.84      0.85       316
           3       0.81      0.88      0.85       368
           4       0.69      0.71      0.70       249
           5       0.77      0.86      0.81       329
           6       0.58      0.65      0.61       186
           7       0.83      0.26      0.40        38
           8       0.78      0.54      0.64        96
           9       1.00      0.33      0.50         6
    accuracy                           0.80      2110
   macro avg       0.80      0.65      0.69      2110
weighted avg       0.80      0.80      0.79      2110
used time: 2.7856316566467285
```

## 二，项目介绍说明

> 新学的指令tree /f,方便快捷^_^

```bash
│  method_comparison.py  # 训练文本分类模型文件
│  README.md 
│  requirements.txt
│  W2V_train.py  # 训练词向量文件（也可以用预训练模型替换词向量）
├─data
│      tianchi_data.csv
├─output
│      model.w2v
```

