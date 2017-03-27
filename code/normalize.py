# -*- coding:utf-8 -*-
# normalize the data to [0,1]
fin = open('../data/out.txt', 'r')
fout = open('../data/out_normalize.txt', 'w')
fout_label = open('../data/label.txt', 'w')

min_max=[]

time=[]
dapan_open=[]
dapan_high=[]
dapan_low=[]
dapan_close=[]
dapan_zhangdie=[]
dapan_zhangfu=[]
dapan_zhenfu=[]
minsheng_open=[]
minsheng_high=[]
minsheng_low=[]
minsheng_close=[]
minsheng_zhangfu=[]
minsheng_zhenfu=[]
minsheng_zongshou=[]
minsheng_jine=[]
minsheng_huanshou=[]
minsheng_chengjiao=[]
fin_lines=fin.readlines()
fout.write(fin_lines[0])
print fin_lines[1].strip('\n').split('\t')
for i in range(1, len(fin_lines)):
    a=fin_lines[i].strip('\n').split('\t')
    time.append(a[0])
    dapan_open.append(float(a[1]))
    dapan_high.append(float(a[2]))
    dapan_low.append(float(a[3]))
    dapan_close.append(float(a[4]))
    dapan_zhangdie.append(float(a[5]))
    dapan_zhangfu.append(float(a[6]))
    dapan_zhenfu.append(float(a[7]))
    minsheng_open.append(float(a[8]))
    minsheng_high.append(float(a[9]))
    minsheng_low.append(float(a[10]))
    minsheng_close.append(float(a[11]))
    minsheng_zhangfu.append(float(a[12]))
    minsheng_zhenfu.append(float(a[13]))
    minsheng_zongshou.append(float(a[14].replace(',','')))
    minsheng_jine.append(float(a[15].replace(',','')))
    minsheng_huanshou.append(float(a[16]))
    minsheng_chengjiao.append(float(a[17]))

def normalize(arr):
    mi=ma=arr[0]
    for i in arr:
        if i<mi:
            mi=i
        if i>ma:
            ma=i
    for i in range(len(arr)):
        arr[i] = 1.0*(arr[i]-mi)/(ma-mi)

def normalize_label(arr):
    mi=ma=arr[0]
    for i in arr:
        if i<mi:
            mi=i
        if i>ma:
            ma=i
    for i in range(len(arr)):
        arr[i] = 1.0*(arr[i]-mi)/(ma-mi)
    # record the min and max to transfer predict_values into stock prices
    min_max.append(mi)
    min_max.append(ma)

normalize(dapan_open)
normalize(dapan_high)
normalize(dapan_low)
normalize(dapan_close)
normalize(dapan_zhangdie)
normalize(dapan_zhangfu)
normalize(dapan_zhenfu)
normalize(minsheng_open)
normalize_label(minsheng_high)
normalize(minsheng_low)
normalize(minsheng_close)
normalize(minsheng_zhangfu)
normalize(minsheng_zhenfu)
normalize(minsheng_zongshou)
normalize(minsheng_jine)
normalize(minsheng_huanshou)
normalize(minsheng_chengjiao)

for i in range(len(dapan_open)):
    fout.write(time[i]+'\t'+str(dapan_open[i])+'\t'+str(dapan_high[i])+'\t'+str(dapan_low[i])+'\t'+str(dapan_close[i])+'\t'+str(dapan_zhangdie[i])+'\t'+str(dapan_zhangfu[i])+'\t'+str(dapan_zhenfu[i])+'\t')
    fout.write(str(minsheng_open[i])+'\t'+str(minsheng_high[i])+'\t'+str(+minsheng_low[i])+'\t'+str(minsheng_close[i])+'\t'+str(minsheng_zhangfu[i])+'\t'+str(minsheng_zhenfu[i])+'\t'+str(minsheng_zongshou[i])+'\t'+str(minsheng_jine[i])+'\t'+str(minsheng_huanshou[i])+'\t'+str(minsheng_chengjiao[i])+'\n')
    fout_label.write(str(minsheng_high[i])+'\n')

fout_label.write('min:'+str(min_max[0])+'\t'+'max:'+str(min_max[1])+'\n')