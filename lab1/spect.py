import numpy as np
import pandas as pd
import random

Max=1000
data=pd.read_csv('SPECT.csv')
copy=data.iloc[:,:]
l=[]
z=[]
ltest=[]
ztest=[]
dict={}
c=0
n=data.iloc[:,1].count()
for i in range(n):
    row=random.randint(0,copy.iloc[:,1].count()-1)
    if i<0.9*n:
        l.append(copy.iloc[row,1:])
        z.append(copy.iloc[row,0])
        if copy.iloc[row,0] not in dict:
            dict[copy.iloc[row,0]]=c
            c+=1
        copy.drop(copy.index[row])
    else:
        ltest.append(copy.iloc[row,1:])
        ztest.append(copy.iloc[row,0])
        copy.drop(copy.index[row])

w=[]
e=[None]*(data.iloc[:,1].count()) #errors
y=e[:]
for i in range(data.iloc[1,:-1].count()): #weights
    w.append(1/(data.iloc[1,:-1].count()+1))
    
for i in range(len(z)): #changing categorical variables to numbers
    z[i]=dict[z[i]]
for i in range(len(ztest)):
    ztest[i]=dict[ztest[i]]

flag=1
for i in range(Max):
    for i in range(len(l)):
        sum=-0.5
        for j in range(len(l[i])):
            sum+=(l[i][j]*w[j])
        if sum>0:
            y[i]=1
        else:
            y[i]=0
        e[i]=z[i]-y[i]

        d=0.1*e[i]
        for j in range(len(l[i])):
            w[j]+=(d*l[i][j])
        flag=0
        for i in e:
            if i==1 or i==-1:
                flag=1
                break
    if flag==1:
        break
ytest=[]
acc=0
tp=0
tn=0
fp=0
fn=0
for i in range(len(ltest)):
    sum=-0.5
    for j in range(len(ltest[i])):
        sum+=(ltest[i][j]*w[j])
    if sum>0:
        ytest.append(1)
    else:
        ytest.append(0)
    if ztest[i]==ytest[-1]:
        if ztest[i]==1:
            tp+=1
        if ztest[i]==0:
            tn+=1
        acc+=1
    else:
        if ytest[-1]==1:
            fp+=1
        else:
            fn+=1

print('accuracy=',(acc/len(ytest))*100)
print('precision(+) =',tp/(tp+fp))
print('recall(+) =',tp/(tp+fn))
print('precision(-) =',tn/(tn+fn))
print('recall(-) =',tn/(tn+fp))