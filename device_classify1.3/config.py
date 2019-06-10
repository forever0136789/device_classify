import os

#源数据路径
path = './data'

# 数据集路径
data_file = './data/ip_info.csv'

# 预处理后的数据集
proc_data_file = os.path.join('./data/proc_device.csv')

# 均衡后的预处理数据集
dataset_file = os.path.join('./data/dataset.csv')

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
        'switch': 12,
        'print server': 13,
        'security-misc': 14,
        'remote management': 15,
        'phone': 16,
        'printer': 17,
        'bridge': 18,
        'VoIP adapter': 19,
        'firewall':20,
        'WAP': 21,
        'general purpose': 22,
        'specialized': 23,
                  }