# -*- coding:utf-8 -*-

# this .py can reduce the dimension of the input vectors
from sklearn.decomposition import PCA
pca=PCA(n_components='mle')

fin = open('out.txt', 'r').readlines()
result=[]
for j in range(1,len(fin)):
    a=fin[j].strip('\n').split('\t')
    tmp=[]
    for i in range(1, len(a)):
        tmp.append(float(a[i].replace(',','')))
    result.append(tmp)


b=pca.fit_transform(result)
print len(b[0])
# fout = open('after_pca.txt', 'w')
# for i in range(len(b)):
#     for j in range(len(b[i])):
#         fout.write(str(b[i])+'\t')
#     fout.write('\n')
