import config

import os
import pandas as pd
import joblib
import numpy as np
import pymysql.cursors

def predict2mysql(ip_dev_reg_dic):
    connection = pymysql.connect(host='localhost', user='wss', password='wss123456', db='wss_db', charset='utf8',cursorclass=pymysql.cursors.DictCursor)
    try:
        # 从数据库链接中得到cursor的数据结构
        with connection.cursor() as cursor:
            # 在之前建立的user表格基础上，插入新数据，这里使用了一个预编译的小技巧，避免每次都要重复写sql的语句
            for ip, dev_reg in ip_dev_reg_dic.items():
                print('{}预测类别{}更新到数据库'.format(ip, dev_reg))
                sql = "UPDATE drose_ipaddress SET dev_reg=%s where ip=%s"
                cursor.execute(sql, (dev_reg, ip))
            # 执行到这一行指令时才是真正改变了数据库，之前只是缓存在内存中
            connection.commit()
    # 最后关闭连接
    finally:
        connection.close()

def predict_type(n=500):#n代表每次预测的数据行数
    reverse_device_type_dict = {value: key for key, value in config.device_type_dict.items()}
    reader = pd.read_csv(config.proc_data_file, chunksize=n)
    i=0
    for test_data in reader:
        #print(test_data.shape)
        y_test = test_data['label'].values
        # 对测试集构建特征矩阵
        test_proc_text = test_data['proc_info'].values
        tfidf_vectorizer = joblib.load(os.path.join(config.output_path,'tfidf_vectorizer_proc_info.m'))
        test_tfidf_feat = tfidf_vectorizer.transform(test_proc_text).toarray()

        test_proc_os_type = test_data['proc_os_type'].values
        tfidf_vectorizer_os = joblib.load(os.path.join(config.output_path,'tfidf_vectorizer_os.m'))
        test_tfidf_os_type = tfidf_vectorizer_os.transform(test_proc_os_type).toarray()
        X_test = np.hstack((test_tfidf_feat, test_tfidf_os_type))

        clf = joblib.load(os.path.join(config.output_path, 'best_model.m'))
        score = clf.score(X_test, y_test)
        i+=1
        print('第{}~{}个数据的测试准确率：{:.3f}'.format((i - 1) * 500 + 1, (i - 1) * 500 + test_data.shape[0], score))

        test_data['dev_reg'] = clf.predict(X_test)
        test_data['dev_reg'] = test_data['dev_reg'].map(reverse_device_type_dict)
        ip_dev_reg_dic = dict(zip(test_data['ip'].values, test_data['dev_reg'].values))
        predict2mysql(ip_dev_reg_dic)
        #预测结果存入.csv
        test_data.to_csv(config.predict_proc_data_file, mode='a', index=False)
        #存储预测错误数据
        test_data[test_data['device'] != test_data['dev_reg']].to_csv(config.uncorrect_predict_data_file, mode='a', index=False)
