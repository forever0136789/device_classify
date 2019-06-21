
import buildclassifier
import config
import predicttype
import es2csv

import os
import pandas as pd

def main():
    if not os.path.exists(config.data_file):
        es2csv.msql2csv()
        es2csv.csv_write(config.data_file)

    if not os.path.exists(os.path.join(config.output_path, 'best_model.m')):
        print('开始构建分类模型...')
        buildclassifier.build_classifier()

    print('分类模型已构建，进入预测阶段...')
    predicttype.predict_type()
    print('预测完成.')

    #预测错误率
    error=pd.read_csv(config.uncorrect_predict_data_file).shape[0]/pd.read_csv(config.predict_proc_data_file).shape[0]
    print('预测错误率：{:.3f}'.format(error))
    if error>0.4:
        print('预测错误率高于0.4,请优化分类模型，重新训练.')


if __name__ == '__main__':
    main()