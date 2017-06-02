#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 22:54:10 2017

@author: dustin
"""
import jieba
import jieba.analyse
import matplotlib.pyplot as plt
from wordcloud import WordCloud
#####读取文件函数
def t(path,file,encoding='utf8'):
    f=open(path+file,'r',encoding=encoding)
    f.seek(0)
    a=f.readlines()
    f.close
    b=' '.join(a)
    return(b)

######获取文本
txt=' '
path='/Users/dustin/Github/Python/jieba/'
for i in range(1,12):
    file=str(i)+'.txt'
    txt=txt+t(path,file)

######分词
jieba.analyse.set_stop_words("/Users/dustin/Github/Python/jieba/stop_words.txt")
t2=jieba.analyse.extract_tags(txt, topK=50, withWeight=True, allowPOS=())
t3=dict(t2)

######词云绘制
cloud = WordCloud(font_path=path+'Lantinghei.ttc',height=1000,width=1000,background_color='white',max_font_size=200)
wc = cloud.generate_from_frequencies(t3)# 产生词云
plt.imshow(wc)
wc.to_file(path+"wc.jpg")

