# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 18:49:55 2017

@author: Hjx
"""

import requests
import time


def file_read(path):
    """
    创建函数读取坐标点txt文件
    输出一个经纬度列表
    path: 文件路径
    """
    f = open(path, 'r')
    text = []
    for i in f.readlines():
        i_ = i[:-2]
        lng = i_.split(',')[0]
        lat = i_.split(',')[1]
        text.append(lat + ',' + lng)
    # print(text)
    return text


def get_params(s,e,c,k):
    '''
    创建网页参数获取函数
    输出参字典列表
    s: 起点经纬度字典列表
    e: 终点经纬度字典列表
    c: 城市名称
    k: 密钥
    '''
    p = []
    s_num = 0
    e_num = 0
    for i in s:
        s_num += 1
        for j in e:
            e_num +=1
            params = {
                    'mode': 'driving',
                    'origin': i,
                    'destination': j,
                    'origin_region': c,
                    'destination_region': c,
                    'output': 'json',
                    'ak': k
                    }
            p.append([params,s_num,e_num])
    return(p)

    
def get_url(u, p):
    """
    创建网页信息请求函数
    输出网页返回信息
    u: 网址url
    p: 参数
    """
    r = requests.get(u, p)
    # print(r.url)
    return r.json()

    
def get_data1(js):
    """
    创建路径距离/时间获取函数
    输出一个字典，结果包括该条路径的总距离、总时间以及路段数量
    """
    result_ = js['result']
    routes_ = result_['routes'][0]
    distance_ = routes_['distance']
    duration_ = routes_['duration']
    num = len(routes_['steps'])
    data_dic = dict([['dis', distance_], ['time', duration_], ['num', num]])
    # print(data_dic)
    return data_dic


def get_data2(js, n):
    """
    创建路径节点获取函数
    输出为一个字典列表，包括每一个节点的经度纬度
    """
    result_ = js['result']
    routes_ = result_['routes'][0]
    steps_ = routes_['steps']
    step = steps_[n]
    duration = step['duration']
    path_points = step['path'].split(';')
    point_lst = []
    for point in path_points[::5]:
        lng = point.split(',')[0]
        lat = point.split(',')[1]
        point_geo = dict([['lng', lng], ['lat', lat],['duration',duration]])
        point_lst.append(point_geo)
    # print(point_lst)
    return (point_lst)

    
def main():
    # 密钥，需要自己填写！
    keys = 'IIfkCB7Brjl0bDhxIcPCCQ68QPAceBGj'
    
    # 网址，不包括参数
    url = 'http://api.map.baidu.com/direction/v1'
    
    # 文件路径，需要自己填写！
    path = '/Users/dustin/Documents/精进/Github/Python/数据及爬虫/'
    
    # 爬取数据所在城市
    city = '上海'
    
    # 调用函数，分别输出起点、终点的经纬度
    start_point = file_read(path + 'start.txt')
    end_point = file_read(path + 'end.txt')
    
    # 创建结果txt文件，并填写标签
    f_result = open(path + 'result.txt', 'w')
    f_result.seek(0)
    f_result.write('路径编号,起点编号,终点编号,节点经度,节点纬度,时间戳\n')
    
    # 设置爬取开始时间
    start_time = time.time()
    
    # 起点编号记录
    pathID = 0
   
    # 获取所有起点、终点的参数
    p = get_params(start_point,end_point,city,keys)
    # print(p)

    for p,sn,en in p:
        pathID += 1
        r_js = get_url(url,p)
        data1 = get_data1(r_js)
        path_num = data1['num']
        point_time = start_time
        for m in range(path_num):
            points = get_data2(r_js, m)
            point_num = len(points)
            # 计算出以i为起点，j为终点的路径中，每个节点之间的平均时间
            for point in points:
                time_per = point['duration'] / point_num
                point_time = point_time + time_per
                # 算出每个节点的时间点
                point_time2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(point_time))
                # 写入数据
                f_result.writelines([
                                     str(pathID),',',
                                     str(sn),',',
                                     str(en),',',
                                     str(point['lng']),',',
                                     str(point['lat']),',',
                                     point_time2,'\n'
                                     ])
    f_result.close()


if __name__ == '__main__':
    main()
    print('finished!')
