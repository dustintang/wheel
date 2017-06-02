# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#excel连接
import xlrd
data=xlrd.open_workbook('/Users/dustin/Desktop/联合办公监测.xlsx')
table=data.sheet_by_index(0)
nrows=table.nrows
ncols=table.ncols
print(table.col_values(0))

import requests,json
from bs4 import BeautifulSoup
r=requests.get(url='http://www.gaohaipeng.com/2251.html')
print(r)
print(r.encoding)
print(r.text)
print(r.headers)
#requests.get/post(url='')，网址？之后对url传递参数用params=字典
#参数timeout设置超时
#抓取
soup=BeautifulSoup(r.text,'lxml')
print(soup.prettify()) #有缩进
print(soup.p) #第一个p标签
print(soup.p.name) #字符串
print(soup.a.attrs['href']) #字典
print(soup.title.string) #特殊格式
print(soup.title.text) #字符串，只会显示内容，不会显示标签等内容

#看子节点
print(soup.head.contents) #列表
print(soup.head.children) #生成器，可以遍历
print(soup.head.descendants)#子孙节点
print(soup.p.parent)#父节点 +s全部父节点可以遍历
#兄弟节点：next_sibling   previous_sibling  +s全部兄弟节点，也是生成器可遍历
print(soup.find_all('p'))#返回特殊的列表格式，找到所有符合的tag。可以提取text、attrs
#find_all(name,attrs,recursive,string,**keywords)
#limit属性返回几个；keyword参数设置:height=20,src=True,re.compile('')
print(soup.find('p',id='1').text)


#json
r=requests.get(url='http://echarts.baidu.com/data/asset/data/confidence-band.json')
print(type(r.json()))#返回列表
print(type(r.text))
data=json.loads(r.text)
js=json.dumps(data)#js是str格式

#json写入文件
f=open('/Users/dustin/Desktop/js.txt','w')
f.seek(0)
f.writelines(['date,u,l,value\n'])

for i in data:
    f.writelines([str(i['date']),',',str(i['u']),',',str(i['l']),',',str(i['value']),'\n'])

f.close()




