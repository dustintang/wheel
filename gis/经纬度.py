#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 20:13:45 2017

@author: dustin
"""
import pymysql,requests
#连接数据库
conn = pymysql.connect(host='127.0.0.1',
                       port=3306, 
                       user='root', 
                       passwd='123456', 
                       db='创邑监测',
                       charset='utf8')
cur = conn.cursor()
#读取地址
cur.execute("SELECT 地址 FROM beijing")
r = cur.fetchall()
#print(cur.description)
url='http://api.map.baidu.com/geocoder/v2/'
for i in r:
    #address=i[2]
    #address_all=str(i[0])+str(i[1])+i[2]
    address = i[0]
    try:
        #geocoding   
        params = {
                'address':address,
                'output':'json',
                'ak':'IIfkCB7Brjl0bDhxIcPCCQ68QPAceBGj',
                }
        r=requests.get(url,params)
        print(r.text)
        r_js=r.json()
        lat = r_js['result']['location']['lat']
        lng = r_js['result']['location']['lng']
        sql1 = 'update beijing set lat="'+str(lat)+'"where 地址 ="'+address+'"'
        sql2 = 'update beijing set lng="'+str(lng)+'"where 地址 ="'+address+'"'
        cur.execute(sql1)
        cur.execute(sql2)
    except:
        continue

#坐标转换————————————————————————
import sys
sys.path.append("/Users/dustin/Documents/study/Github/Python/wheel/function")
import coord

cur.execute("SELECT lng,lat FROM beijing")
r = cur.fetchall()
for i in r:
    if i[0] != None:
        a=coord.bd09togcj02(i[0],i[1])
        b=coord.gcj02towgs84(a[0],a[1])
        sql1 = 'update beijing set wgs84_lng="'+str(b[0])+'" where lng ="'+str(i[0])+'"and lat ="'+str(i[1])+'"'
        sql2 = 'update beijing set wgs84_lat="'+str(b[1])+'" where lng ="'+str(i[0])+'"and lat ="'+str(i[1])+'"'
        cur.execute(sql1)
        cur.execute(sql2)
        print(b)
    


#提交修改
conn.commit()
#关闭连接
cur.close()
conn.close()

