# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:00:59 2015

@author: admin
"""
import pandas as pd
import scipy as sp

filename='C:\Python27\data\preference.csv'
 
reader=pd.read_csv(filename,iterator=True)

loop = True
chunkSize = 100000
chunks = [] 
while loop: 
    try:    
      chunk = reader.get_chunk(chunkSize) 
      chunks.append(chunk)  
    except StopIteration:    
        loop = False  
        print "Iteration is stopped."
df = pd.concat(chunks, ignore_index=True) 





'''
f=open(filename)
count=0
for line in f:
    count+=1
    print line 
    if count==10:
      break
'''
