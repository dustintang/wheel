#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:35:17 2017

@author: dustin
"""
import pymysql
conn = pymysql.connect(host='127.0.0.1',
                       port=3306, 
                       user='root', 
                       passwd='123456', 
                       db='创邑监测',
                       charset='utf8')
cur = conn.cursor()
#读取地址，统一地址信息————————————————————
cur.execute("SELECT 品牌,空间名,地址 FROM 空间")
r = cur.fetchall()
for i in r:
    a=i[2].split('区',1)
    try:
        sql = 'update 空间 set 地址="'+a[1]+'" where 品牌 ="'+i[0]+'"and 空间名 ="'+i[1]+'"'
    except:
        sql = 'update 空间 set 地址="'+a[0]+'" where 品牌 ="'+i[0]+'"and 空间名 ="'+i[1]+'"'
    cur.execute(sql)  



#提交修改——————————————————————
conn.commit()
#关闭连接
cur.close()
conn.close()
