import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score


def main():
    """
        主函数
    """
    # 准备数据集
    preprocess.origin_data()
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
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)

    print('准确率：', accuracy_score(y_test, y_pred))
    print('AUC值：', roc_auc_score(y_test, y_pred))#类别分布不均常用AUC评价
    #对于类别不均衡的样本集解决方法有：对类别中样本多的进行抽样、对类别中样本少的深采样、增加样本少的类别的权重（类似于adaboost）。

    #使用词袋模型+tfidf、贝叶斯分类:共有3848维特征。准确率： 0.7058823529411765。AUC值： 0.6944444444444444

if __name__ == '__main__':
    main()
