# -*- coding:utf-8 -*-

#convert the raw_data into semi_raw_data
from matplotlib.pyplot import *
file1 = open('../data/minsheng.txt', 'r')
file2 = open('../data/dapan_chuliguo.txt', 'r')
fout = open('../data/out.txt','w')
fout_norm = open('../data/out_normalize.txt', 'w')

dapan=[]
minsheng=[]
x=[]
labels=[]
dapan_lines=file2.readlines()
minsheng_lines=file1.readlines()
a=dapan_lines[1].replace('\t',' ').replace('  ',' ').split(' ')
#print dapan_lines[1].replace('\t',' ').replace('  ',' ').split(' ')
fout.write('时间'+'\t'+'大盘开盘'+'\t'+'大盘最高'+'\t'+'大盘最低'+'\t'+'大盘收盘'+'\t'+ '大盘涨跌'+'\t'+'大盘涨幅'+'\t'+'大盘振幅'+'\t'+'开盘'+'\t'+'最高'+'\t'+'最低'+'\t'+'收盘'+'\t'+'涨幅'+'\t'+'振幅'+'\t'+'总手'+'\t'+'金额'+'\t'+'换手'+'\t'+'成交次数'+'\n')
fout_norm.write('时间'+'\t'+'大盘开盘'+'\t'+'大盘最高'+'\t'+'大盘最低'+'\t'+'大盘收盘'+'\t'+ '大盘涨跌'+'\t'+'大盘涨幅'+'\t'+'大盘振幅'+'\t'+'开盘'+'\t'+'最高'+'\t'+'最低'+'\t'+'收盘'+'\t'+'涨幅'+'\t'+'振幅'+'\t'+'总手'+'\t'+'金额'+'\t'+'换手'+'\t'+'成交次数'+'\n')
#print a[0][:10]
for i in range(1,len(dapan_lines)-1):
    a=dapan_lines[i].replace('\t',' ').replace('  ',' ').split(' ')
    b=minsheng_lines[i].replace('\t',' ').replace('  ',' ').split(' ')
    fout.write(a[0][:10]+'\t')
    for j in range(1,len(a)-3):
        fout.write(a[j]+'\t')
    for j in range(1,len(b)-2):
        fout.write(b[j].strip('%')+'\t')
    fout.write(b[len(b)-2]+'\n')

# for line in dapan_lines:
#     print line.replace('\t','').split(' ')
#for i in range(6166, len(lines)):
    