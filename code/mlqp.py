from matplotlib.pyplot import *
from numpy.random import *
from math import *
N=[85,11,1];M=len(N)
lr_u=0.08;lr_v=0.08;lr_b=0.08;nb=200000;smooth=1
u=[];v=[];b=[];x=[];x_=[];de_dx=[] #x_ is net, x is output, de_dex is deltas
s_u=[];s_v=[];s_b=[]
do_not_use_u=1
use_entropy=1

train_set = open('../data/train.txt','r').readlines()
data=[]
for i in train_set:
	tmp=[]
	a=i.strip('\t\n').split('\t')
	for j in a:
		tmp.append(float(j))
	data.append(tmp)
test_set = open('../data/test.txt','r').readlines()
test=[]
for i in test_set:
	tmp=[]
	a=i.strip('\t\n').split('\t')
	for j in a:
		tmp.append(float(j))
	test.append(tmp)
fout = open('../data/predict.txt','w')


print("read complete!")

def init(n,m):
	t=[[0]*m for i in range(n)]
	for i in range(n):
		for j in range(m):
			t[i][j]=randn(1)[0]
	return t

def zero_(n,m):
	t=[[0]*m for i in range(n)]
	for i in range(n):
		for j in range(m):
			t[i][j]=0
	return t

#sigmoid
def f(x):
	return 1/(1+exp(-x))

#sigmoid'
def f_(x):
	return f(x)*(1-f(x))

for i in range(M):
	if i!=0:
		u.append(init(N[i],N[i-1]))
		v.append(init(N[i],N[i-1]))
		s_u.append(zero_(N[i],N[i-1]))
		s_v.append(zero_(N[i],N[i-1]))
	else:
		u.append([])
		v.append([])
		s_u.append([])
		s_v.append([])
	b.append([randn(1)[0]]*N[i])
	s_b.append([0]*N[i])
	x.append([0]*N[i])
	x_.append([0]*N[i])
	de_dx.append([0]*N[i])
history=[]
history2=[]
if smooth==1:
	nb-=nb%len(data)
#print(u)
#print(v)
#print(b)
print("complete init!")

def forward(data_):
	#forward pass
	for i in range(N[0]):
		x[0][i]=data_[i]
	for i in range(1,M):
		for j in range(N[i]):
			x_[i][j]=b[i][j]
			for k in range(N[i-1]):
				if do_not_use_u==1:
					x_[i][j]+=v[i][j][k]*x[i-1][k]
				else:
					x_[i][j]+=u[i][j][k]*x[i-1][k]**2+v[i][j][k]*x[i-1][k]
			x[i][j]=f(x_[i][j])
	return x[-1][0]
print("start train...")			
for m in range(nb):
	n=m#randint(0,len(data))
	forward(data[n%len(data)])
	#print(x_)
	#print(x)
	#calculate error
	e=0
	for i in range(N[-1]):
		e+=(x[-1][i]-data[n%len(data)][N[0]+i])**2
	e/=2
	history+=[e]
	#if m%(nb//min(1000,nb))==0 or m==nb-1:
		#print("iteration "+str(m)+', error '+str(e))
	if m%len(data)==len(data)-1 and smooth==1:
		e=0
		for i in range(len(data)):
			e+=history[-1]
			history.pop(-1)
		e/=len(data)
		history.append(e)
		history2.append(m+1)
		if len(history)%30==0:
			print("iteration "+str(m)+', error '+str(e))
	#backward pass
	for i in range(1,M)[::-1]:
		for j in range(N[i]):
			if i==M-1:
				de_dx[i][j]=x[i][j]-data[n%len(data)][N[0]+j]
			else:
				de_dx[i][j]=0
				for k in range(N[i+1]):
					if i+1==M-1 and use_entropy:
						#cross entropy
						de_dx[i][j]+=de_dx[i+1][k]*(2*u[i+1][k][j]*x[i][j]+v[i+1][k][j])
					else:
						de_dx[i][j]+=de_dx[i+1][k]*f_(x_[i+1][k])*(2*u[i+1][k][j]*x[i][j]+v[i+1][k][j])
	for i in range(1,M):
		for j in range(N[i]):
			for k in range(N[i-1]):
				if i==M-1 and use_entropy:
					s_u[i][j][k]=s_u[i][j][k]*0.5-lr_u*de_dx[i][j]*x[i-1][k]**2
					s_v[i][j][k]=s_v[i][j][k]*0.5-lr_v*de_dx[i][j]*x[i-1][k]
				else:
					s_u[i][j][k]=s_u[i][j][k]*0.5-lr_u*de_dx[i][j]*f_(x_[i][j])*x[i-1][k]**2
					s_v[i][j][k]=s_v[i][j][k]*0.5-lr_v*de_dx[i][j]*f_(x_[i][j])*x[i-1][k]
				u[i][j][k]+=s_u[i][j][k]
				v[i][j][k]+=s_v[i][j][k]
			if i==M-1 and use_entropy:
				s_b[i][j]=s_b[i][j]*0.5-lr_b*de_dx[i][j]
			else:
				s_b[i][j]=s_b[i][j]*0.5-lr_b*f_(x_[i][j])*de_dx[i][j]
			b[i][j]+=s_b[i][j]
print("start predict...")
for case in test:
	fout.write(str(forward(case))+'\n')




# title("learning rate="+str(lr_b))
# xlabel("number of iterations")
# ylabel("error")
# plot(history2,history)
# savefig("history.png")
# cla()
# zero_points=[]
# one_points=[]
# r=100
# for i in range(r+1):
# 	for j in range(r+1):
# 		forward([i/r,j/r])
# 		if x[-1][0]>0.5:
# 			one_points.append([i/r,j/r])
# 		else:
# 			zero_points.append([i/r,j/r])
# figure(figsize=(10,10))
# xticks([0,0.5,1])
# yticks([0,0.5,1])
# scatter([i[0]for i in zero_points], [i[1]for i in zero_points], color='blue',s = 50)
# scatter([i[0]for i in one_points], [i[1]for i in one_points], color='red',s = 50)
# scatter([i[0]for i in data[::2]], [i[1]for i in data[::2]], marker='x',color='black',alpha=1)
# scatter([i[0]for i in data[1::2]], [i[1]for i in data[1::2]], marker='o',color='black',alpha=1)
# savefig('result.png')