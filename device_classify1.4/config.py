import os

#源数据路径
path = './data'

#
ipaddress_file='./data/ip_info.csv'

# 数据集路径
data_file = './data/ip_info.csv'

# 预处理后的数据集
proc_data_file = os.path.join('./data/proc_device.csv')

# 均衡后的预处理数据集
dataset_file = os.path.join('./data/dataset.csv')

#预测完成后的数据集
predict_proc_data_file=os.path.join('./data/predict_proc_device.csv')
#预测错误的数据集
uncorrect_predict_data_file=os.path.join('./data/uncorrect_predict_data.csv')

output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 文本类别字典
device_type_dict = {
        'router': 0,
        'webcam': 1,
        'PBX': 2,
        'broadband router': 3,
        'media device': 4,
        'proxy server': 5,
        'load balancer': 6,
        'power-device': 7,
        'game console': 8,
        'storage-misc': 9,
        'VoIP phone': 10,
        'terminal': 11,
        'terminal server':12,
        'switch': 13,
        'print server': 14,
        'security-misc': 15,
        'remote management': 16,
        'phone': 17,
        'printer': 18,
        'bridge': 19,
        'VoIP adapter': 20,
        'firewall':21,
        'WAP': 22,
        'general purpose': 23,
        'specialized': 24
                  }