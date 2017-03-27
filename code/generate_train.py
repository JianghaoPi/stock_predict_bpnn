# -*- coding:utf-8 -*-

#convert the semi_raw_data into train and test sets
from sklearn.decomposition import PCA

pca=PCA(n_components=15)

fin = open('../data/out_normalize.txt','r')
fin_label=open('../data/label.txt', 'r').readlines()
fout_train = open('../data/train.txt', 'w')
fout_test = open('../data/test.txt', 'w')
fin_lines = fin.readlines()
highD_data=[]
for i in range(1,len(fin_lines)-5):
    tmp=[]
    for k in range(5):
        a=fin_lines[i+k].replace('\n', '').split('\t')
        for j in range(1, len(a)):
            tmp.append(a[j])
    b = fin_label[i+4].replace('\n','')
    tmp.append(b)
    highD_data.append(tmp)

# result = pca.fit_transform(highD_data)
# print len(result[0])
# count = 0
# for i in pca.explained_variance_ratio_:
#     count += i
# print count
for i in range(201):
    for j in range(len(highD_data[i])):
        fout_train.write(str(highD_data[i][j])+'\t')
    fout_train.write('\n')
for i in range(201,len(highD_data)):
    for j in range(len(highD_data[i])):
        fout_test.write(str(highD_data[i][j])+'\t')
    fout_test.write('\n')
