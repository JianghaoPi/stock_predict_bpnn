# test the functions

train_set = open('../data/train.txt','r').readlines()
data=[]
for i in train_set:
	tmp=[]
	a=i.strip('\t\n').split('\t')
	for j in a:
		tmp.append(float(j))
	data.append(tmp)

print data[0][85]
