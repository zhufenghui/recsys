# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 08:53:39 2015

@author: admin
"""

import pandas as pd 
import numpy as np 
import os 
import codecs

path='C:/Python27/pyfiles/recsys/BX-Dump/BX-Book-Ratings.csv'

def loadData(path):
    f=codecs.open(path)
    lines=[]
    data={}
    for line in f:
        line = line.split(";")
        user=line[0].strip('"')
        book=line[1].strip('"')
        rating=line[2].strip().strip('"') #先后有区别
        item=(user,book,rating)
        lines.append(item)
        if user in data:
            book_rating=data[user]  # 将行字典赋予用户
        else:
            book_rating={}
        book_rating[book]=rating
        data[user]=book_rating
        
    return data
   
data=loadData(path) 

