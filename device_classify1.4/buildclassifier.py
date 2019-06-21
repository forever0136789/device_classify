import preprocess
import config

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import GridSearchCV
import joblib

def train_model(X_train, y_train, X_test, y_test, model_name, model, param_range):
    """

        根据给定的参数训练模型，并返回
        1. 最优模型
        2. 平均训练耗时
        3. 准确率
    """
    print('训练{}...'.format(model_name))
    clf = GridSearchCV(estimator=model,
                       param_grid=param_range,
                       cv=5,
                       scoring='accuracy',
                       refit=True)#refit=True,完成五折交叉验证后又进行一次整个训练集的训练
    start = time.time()
    clf.fit(X_train, y_train)
    # 计时
    end = time.time()
    duration = end - start
    print('耗时{:.4f}s'.format(duration))

    # 验证模型
    print('最优参数为：{},验证集最高分为：{}'.format(clf.best_params_,clf.best_score_))
    print('训练准确率：{:.3f}'.format(clf.score(X_train, y_train)))

    score = clf.score(X_test, y_test)
    print('测试准确率：{:.3f}'.format(score))
    print('训练模型耗时: {:.4f}s'.format(duration))
    print()

    return clf, score, duration

def build_classifier():
    """
        得到最佳分类模型
    """
    # 准备数据集
    train_data, test_data = preprocess.prepare_data()

    # 查看数据集
    preprocess.inspect_dataset(train_data, test_data)

    # 特征工程处理
    # 构建训练测试数据
    X_train, X_test = preprocess.do_feature_engineering(train_data, test_data)

    print('共有{}维特征。'.format(X_train.shape[1]))

    # 标签处理
    y_train = train_data['label'].values
    y_test = test_data['label'].values

    # 数据建模及验证
    print('\n===================== 数据建模及验证 =====================')
    sclf = StackingClassifier(classifiers=[KNeighborsClassifier(),
                                           SVC(kernel='linear'),
                                           DecisionTreeClassifier()],
                              meta_classifier=LogisticRegression())
    # 指定各分类器的参数
    model_name_param_dict = {'kNN': (KNeighborsClassifier(), {'n_neighbors': [5, 15, 25]}),
                             'LR': (LogisticRegression(),{'C': [0.01, 1,10,30,50,100]}),
                             'SVM': (SVC(kernel='linear'),{'C': [0.01, 1,30,50,100]}),
                             #'DT': (DecisionTreeClassifier(),{'max_depth': [50,100,150,200,250]}),
                             'RF': (RandomForestClassifier(),{'n_estimators': [50,100, 150, 200, 250]}),
                             'NB': (GaussianNB(), {'priors': [None]}),
                             #'Stacking': (sclf,{'kneighborsclassifier__n_neighbors': [5, 15, 25],'svc__C': [0.01, 1, 100],'decisiontreeclassifier__max_depth': [50, 100, 150],'meta-logisticregression__C': [0.01, 1, 100]}),
                             #'AdaBoost' : (AdaBoostClassifier(),{'n_estimators': [50, 100, 150, 200,250]}),
                             #'GBDT': (GradientBoostingClassifier(),{'learning_rate': [0.01, 0.1, 1, 10, 100]}),
                             }
    # 比较结果的DataFrame
    results_df = pd.DataFrame(columns=['Accuracy (%)', 'Time (s)'],
                              index=list(model_name_param_dict.keys()))
    results_df.index.name = 'Model'

    best_clf_dic={}
    for model_name, (model, param_range) in model_name_param_dict.items():
        best_clf, best_acc, mean_duration = train_model(X_train, y_train, X_test, y_test,
                                                        model_name, model, param_range)
        #存储各算法的最优模型
        best_clf_dic[model_name]=best_clf
        results_df.loc[model_name, 'Accuracy (%)'] = best_acc * 100
        results_df.loc[model_name, 'Time (s)'] = mean_duration
    results_df.to_csv(os.path.join(config.output_path, 'model_comparison.csv'))

    #保存最优模型
    print('最优模型为：',best_clf_dic[results_df['Accuracy (%)'].map(float).idxmax()])
    best_model=best_clf_dic[results_df['Accuracy (%)'].map(float).idxmax()]
    joblib.dump(best_model, os.path.join(config.output_path, 'best_model.m'))

    # 模型及结果比较
    print('\n===================== 模型及结果比较 =====================')

    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    results_df.plot(y=['Accuracy (%)'], kind='bar', ylim=[40, 100], ax=ax1, title='Accuracy(%)', legend=False)

    ax2 = plt.subplot(1, 2, 2)
    results_df.plot(y=['Time (s)'], kind='bar', ax=ax2, title='Time (s)', legend=False)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_path, './pred_results.png'))
    plt.show()

