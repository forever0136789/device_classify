# -*- coding: utf-8 -*-
import config

from elasticsearch import Elasticsearch
import MySQLdb
import csv


def msql2csv():
	db = MySQLdb.connect("localhost",'wss','wss123456','wss_db',charset='utf8')
	cursor = db.cursor()
	cursor.execute("select ip from drose_ipaddress")
	data = cursor.fetchone()
	i=0
	with open(config.ipaddress_file,'w',newline='') as csvfile:
		writer = csv.writer(csvfile)
		while data:
			i=i+1
			if i%5000==0:
				print(i)
			writer.writerow(data)
			data = cursor.fetchone()
	db.close()

#向es按ip查询
def search_es(ip):
    query_dsl = {
        'size': 10000,
        'query': {
            'bool': {
                'must': {'match':{'ip':ip}},
            }
        },
    }
    es = Elasticsearch("127.0.0.1:9200")
    responce = es.search(index="es_scanresult", body=query_dsl)
    #print('搜索耗时：%dms'%responce['took'])
    #print(responce)
    data = responce["hits"]["hits"]
    return data

def ip_data_gener(ip):
    es_data=search_es(ip)
    #print(es_data)
    ip_data={}
    ip_data['port_list']={}
    ip_data['raw_data']=''
    if es_data:
        ip_data['ip']=ip
        if es_data[0]['_source']['os_type']:
            ip_data['os_type']=es_data[0]['_source']['os_type']
        else:
            ip_data['os_type']=''
        if es_data[0]['_source']['device']:
            ip_data['device']=es_data[0]['_source']['device']
        else:
            ip_data['device']=''
        for item in es_data:
            if item['_source']['name'] in ip_data['port_list']:
                ip_data['port_list'][item['_source']['name']].append(item['_source']['port'])
            else:
                ip_data['port_list'][item['_source']['name']]=[item['_source']['port']]
            if item['_source']['response']:
                ip_data['raw_data'] = ip_data['raw_data'] + ' ' + item['_source']['response']
        return ip_data
    else:
        return None
    
def csv_write(csv_file):   
    i=0
    with open(config.ipaddress_file,encoding='utf-8') as csvfile_read:
        reader = csv.reader(csvfile_read)
        with open(csv_file,'w',newline='') as csvfile_write:
            writer = csv.writer(csvfile_write)
            writer.writerow(['ip','port_list','os_type','raw_data','device'])
            for row in reader:   
                if row:
                    ip_data=ip_data_gener(row[0])
                    i=i+1;
                    if ip_data and ip_data['raw_data'] and ip_data['device']:
                        writer.writerow([ip_data['ip'],ip_data['port_list'],ip_data['os_type'],ip_data['raw_data'],ip_data['device']])
                if i%5000==0:
                    print(i)


#data=ip_data_gener('72.24.47.1')
#data=ip_data_gener('121.248.48.1')
#print(data)

#msql2csv()
#csv_write("../data/ip_info_new.csv")