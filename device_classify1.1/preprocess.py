import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import json
import re
import os
import nltk
#nltk.download()

import config

#整理源数据为初始数据集
def origin_data():
    if os.path.exists(config.data_file):
        # 如果存在预处理的数据集，直接读取
        return
    else:
        # 浏览文件夹:os.listdir
        devices = os.listdir(config.path)
        device_info_dic = {}
        dic_list = []
        for device in devices:
            for file in os.listdir(config.path + '/' + device):
                device_info_dic['ip'] = re.findall(r'[0-9]+(?:\.[0-9]+){3}', file)[0]
                device_info_dic['device'] = device

                with open(config.path + '/' + device + '/' + file, 'r') as load_f:
                    load_dict = json.load(load_f)
                # 把每个端口的banner信息作set（）处理
                dic = {port: list(set(load_dict[port])) for port in load_dict.keys()}
                for port in dic.keys():
                    # 对http协议，若有连接成功的进行提取作为banner
                    for j in range(len(dic[port])):
                        if re.match(r'HTTP.*OK', load_dict[port][j]) != None:
                            dic[port] = load_dict[port][j]
                            break
                        elif re.match(r'HTTP.*', load_dict[port][j]) != None:
                            dic[port] = load_dict[port][j]
                        else:
                            continue
                ss = ''
                for key in dic:
                    if isinstance(dic[key], list):
                        dic[key] = ''.join(dic[key])
                    ss += dic[key]
                # print(ss)
                device_info_dic['info'] = ss

                dic_list.append(device_info_dic.copy())
        df = pd.DataFrame(dic_list)
        df.to_csv(config.data_file,index=False)

def preprocess_text(raw_text):
    """
        文本预处理操作
        参数：
            - raw_text  原始文本
        返回：
            - proc_text 处理后的文本
    """
    # 全部转换为小写
    raw_text = raw_text.lower()

    # 1. 使用正则表达式去除标点符号，string.punctuation中包含英文的标点
    filter_pattern = re.compile('[%s]' % re.escape(string.punctuation))
    # .sub替换字符串中匹配项
    words_only = filter_pattern.sub('', raw_text)

    # 2. 分词
    raw_words = nltk.word_tokenize(words_only)

    # 3. 词形归一化
    wordnet_lematizer = WordNetLemmatizer()
    words = [wordnet_lematizer.lemmatize(raw_word) for raw_word in raw_words]

    # 4. 去除停用词
    filtered_words = [word for word in words if word not in stopwords.words('english')]

    proc_text = ' '.join(filtered_words)

    return proc_text

def prepare_data():
    """
        准备数据集
        返回：
            - train_data    训练数据
            - test_data     测试数据
    """
    if os.path.exists(config.proc_data_file):
        # 如果存在预处理的数据集，直接读取
        print('读取预处理结果...')
        all_data = pd.read_csv(config.proc_data_file)
    else:
        # 如果不存在预处理的数据集，需要预处理
        print('文本预处理...')
        all_data = pd.read_csv(config.data_file)

        # 添加标签
        all_data['label'] = all_data['device'].map(config.device_type_dict)

        # 添加预处理后的文本
        all_data['proc_info'] = all_data['info'].apply(preprocess_text)

        # 过滤掉空字符串
        all_data = all_data[all_data['proc_info'] != '']

        # 保存预处理结果
        all_data.to_csv(config.proc_data_file, index=False)

    train_data, test_data = train_test_split(all_data, test_size=1/4, random_state=0)

    return train_data, test_data

def inspect_dataset(train_data, test_data):
    """
        查看数据集
        参数：
            - train_data    训练数据
            - test_data     测试数据
    """
    print('\n===================== 数据查看 =====================')
    print('训练集有{}条记录。'.format(len(train_data)))
    print('测试集有{}条记录。'.format(len(test_data)))

    # 可视化各类别的数量统计图
    plt.figure(figsize=(10, 5))

    # 训练集
    ax1 = plt.subplot(1, 2, 1)
    sns.countplot(x='device', data=train_data)

    plt.title('Training Data')
    plt.xlabel('Type')
    plt.ylabel('Count')

    # 测试集
    plt.subplot(1, 2, 2, sharey=ax1)
    sns.countplot(x='device', data=test_data)

    plt.title('Test Data')
    plt.xlabel('Type')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()


def do_feature_engineering(train_data, test_data):
    """
        特征工程获取文本特征
        参数：
            - train_data    训练样本集
            - test_data     测试样本集
        返回：
            - train_X       训练特征
            - test_X        测试特征
    """

    train_proc_text = train_data['proc_info'].values
    test_proc_text = test_data['proc_info'].values

    # TF-IDF特征提取
    tfidf_vectorizer = TfidfVectorizer(max_features=650)
    train_tfidf_feat = tfidf_vectorizer.fit_transform(train_proc_text).toarray()
    test_tfidf_feat = tfidf_vectorizer.transform(test_proc_text).toarray()
    print('特征名：{}'.format(tfidf_vectorizer.get_feature_names()))
    #print(train_tfidf_feat)
    #print(train_tfidf_feat.shape)

    # 词袋模型
    count_vectorizer = CountVectorizer()
    train_count_feat = count_vectorizer.fit_transform(train_proc_text).toarray()
    testcount_feat = count_vectorizer.transform(test_proc_text).toarray()
    #print(train_count_feat,testcount_feat)
    #print(train_count_feat.shape,testcount_feat.shape)

    # 合并特征
    train_X = np.hstack((train_tfidf_feat, train_count_feat))
    test_X = np.hstack((test_tfidf_feat, testcount_feat))

    #不使用词袋模型仅tfidf、贝叶斯分类：共有1924维特征。准确率： 0.6470588235294118。AUC值： 0.6319444444444444
    #train_X=train_tfidf_feat
    #test_X=test_tfidf_feat

    return train_X, test_X

#origin_data()
#train_data,test_data=prepare_data()



