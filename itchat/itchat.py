#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:04:06 2017

@author: dustin
"""

import itchat

itchat.auto_login()
itchat.send('Hello, filehelper', toUserName='filehelper')
itchat.auto_login(hotReload=True)

itchat.send('Hello, YF', toUserName='13600184189')

#0--------
import itchat, time
from itchat.content import *

@itchat.msg_register([TEXT, MAP, CARD, NOTE, SHARING])
def text_reply(msg):
    itchat.send('%s: %s' % (msg['Type'], msg['Text']), msg['FromUserName'])

@itchat.msg_register([PICTURE, RECORDING, ATTACHMENT, VIDEO])
def download_files(msg):
    msg['Text'](msg['FileName'])
    return '@%s@%s' % ({'Picture': 'img', 'Video': 'vid'}.get(msg['Type'], 'fil'), msg['FileName'])

@itchat.msg_register(FRIENDS)
def add_friend(msg):
    itchat.add_friend(**msg['Text']) # 该操作会自动将新好友的消息录入，不需要重载通讯录
    itchat.send_msg('Nice to meet you!', msg['RecommendInfo']['UserName'])

@itchat.msg_register(TEXT, isGroupChat=True)
def text_reply(msg):
    if msg['isAt']:
        itchat.send(u'@%s\u2005I received: %s' % (msg['ActualNickName'], msg['Content']), msg['FromUserName'])

itchat.auto_login(True)
itchat.run()


#1-----
from collections import Counter

friends = itchat.get_friends(update=True)[0:]# 初始化计数器，有男有女，当然，有些人是不填的
male = female = other = 0# 遍历这个列表，列表里第一位是自己，所以从"自己"之后开始计算# 1表示男性，2女性
for i in friends[1:]:
    sex = i["Sex"]
    if sex == 1:
        male += 1
    elif sex == 2:
        female += 1
    else:
        other += 1# 总数算上，好计算比例啊～
        total = len(friends[1:])# 好了，打印结果
print(u"男性好友：%.2f%%" % (float(male) / total * 100))
print(u"女性好友：%.2f%%" % (float(female) / total * 100))
print(u"其他：%.2f%%" % (float(other) / total * 100))

province = []
for i in friends[1:]:
    province.append(i['Province'])
Counter(province)

#2------
# 获取自己的用户信息，返回自己的属性字典
itchat.search_friends()
# 获取特定UserName的用户信息
itchat.search_friends(userName='帆')
# 获取任何一项等于name键值的用户
itchat.search_friends(name='sail_around')
# 获取分别对应相应键值的用户
itchat.search_friends(wechatAccount='filehelper')
# 三、四项功能可以一同使用
itchat.search_friends(name='LittleCoder机器人', wechatAccount='littlecodersh')



