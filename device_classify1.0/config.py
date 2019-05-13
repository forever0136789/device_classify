import os

#源数据路径
path = './data'

# 数据集路径
data_file = './data/device.csv'

# 预处理后的数据集
proc_data_file = os.path.join('./data/proc_device.csv')


# 文本类别字典
device_type_dict = {'webcam': 1,
                  'router': 0}