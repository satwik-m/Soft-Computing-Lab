import pandas as pd
import numpy as np
import random
import math

data=pd.read_csv('SPECT.csv')
copy=data.iloc[:,:]
ltrain=[]
ztrain=[]
ltest=[]
ztest=[]
c=0
n=data.iloc[:,1].count()

for i in range(n):            # dividing train and test dataset
    row=random.randint(0,copy.iloc[:,1].count()-1)
    if i<0.9*n:
        ltrain.append(copy.iloc[row,1:])
        ztrain.append(copy.iloc[row,0])
        copy.drop(copy.index[row])
    else:
        ltest.append(copy.iloc[row,1:])
        ztest.append(copy.iloc[row,0])
        copy.drop(copy.index[row])
        
dict=[[[0,0] for j in range(2)] for i in range(len(ltest[0]))] #storing count of the variables required for probabilities

total=[0,0]
for i in range(len(ltrain[0])):
    for j in range(len(ltrain)):
        if ztrain[j]=='Yes':
            #print(ltest[i][j])
            dict[i][ltrain[j][i]][1]+=1
            total[1]+=1
        else:
            dict[i][ltrain[j][i]][0]+=1
            total[0]+=1
predictions=[]
#print(total)
tp=0
tn=0
fp=0
fn=0
for i in range(len(ztest)):
    P_yes=1
    P_no=1
    for j in range(len(ltest[i])):
        P_yes*=(dict[j][ltest[i][j]][1]/total[1])
        P_no*=(dict[j][ltest[i][j]][0]/total[0])
    P_yes*=(total[1]/(total[0]+total[1]))
    P_no*=(total[0]/(total[0]+total[1]))
    P_y=P_yes/(P_yes+P_no)
    P_n=P_no/(P_yes+P_no)
    if P_yes>P_no:
        predictions.append('Yes')
        if ztest[i]=='Yes':
            tp+=1
        else:
            fp+=1
    else:
        predictions.append('No')
        if ztest[i]=='No':
            tn+=1
        else:
            fn+=1


print('accuracy =',(tp+tn)/len(ztest)*100,'%')
print('precision(+) =',tp/(tp+fp))
print('recall(+) =',tp/(tp+fn))
print('precision(-) =',tn/(tn+fn))
print('recall(-) =',tn/(tn+fp))

