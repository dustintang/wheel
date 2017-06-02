#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 22:02:33 2017
百度地图api
@author: dustin
"""
import requests
key='IIfkCB7Brjl0bDhxIcPCCQ68QPAceBGj'
service={'poi检索':'http://api.map.baidu.com/place/v2/search',
         'poi详情':'http://api.map.baidu.com/place/v2/detail',
         '线路规划':'http://api.map.baidu.com/direction/v1'}
#1路径规划————————
route_params = {
        'mode':'driving',
        'origin':'上海大学(延长校区)',
        'destination':'人民广场地铁站',
        'origin_region':'上海',
        'destination_region':'上海',
        'output':'json',
        'ak':key
        }
##写入参数，字典格式
r=requests.get(service['线路规划'],route_params)
r_js=r.json()
#返回json数据
routes = r_js['result']['routes'][0]
dis = routes['distance']
time = routes['duration']
#路径基本信息
steps = routes['steps']
path = steps['path']
path_lst = path.split(';')
#路径节点写入文件
f_path = '/Users/dustin/Documents/精进/数据团/practise/result.csv'
f_re = open(f_path,'w')
f_re.write('lng,lat\n')
steps = routes['steps']
for step in steps:
    path = step['path']
    point_lst=path.split(';')
    for point in point_lst:
        lng = point.split(',')[0]
        lat = point.split(',')[1]
        f_re.writelines([str(lng),',',str(lat),'\n'])
f_re.close()