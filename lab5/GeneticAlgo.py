import pandas as pd
import numpy as np
import random
import math

def predict(l,z,total,sel):
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(len(z)):
        P_yes=1
        P_no=1
        for j in range(len(l[i])):
            if sel[j]==1:
                P_yes*=(dict[j][l[i][j]][1]/total[1])
                P_no*=(dict[j][l[i][j]][0]/total[0])
        P_yes*=(total[1]/(total[0]+total[1]))
        P_no*=(total[0]/(total[0]+total[1]))
        P_y=P_yes/(P_yes+P_no)
        P_n=P_no/(P_yes+P_no)
        if P_yes>P_no:
            #predictions.append('Yes')
            if z[i]=='Yes':
                tp+=1
            else:
                fp+=1
        else:
            #predictions.append('No')
            if z[i]=='No':
                tn+=1
            else:
                fn+=1
    return (tp,tn,fp,fn)

data=pd.read_csv('SPECT.csv')
copy=data.iloc[:,:]
ltrain=[]
ztrain=[]
ltest=[]
ztest=[]
c=0
n=data.iloc[:,1].count()
pop_size=30
cross_rate=0.25
mutation=0.1

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
      
popn=[[random.randint(0,1) for i in range(len(ltrain[0]))] for j in range(pop_size)]
temppopn=[None for j in range(pop_size)]
acclist=[None for i in range(30)]
Max=0
maxindex=0
for ite in range(100):
    tempmax=0
    tempmaxindex=0
    totalfitness=0
    cumulfitness=[]
    for sel in range(len(popn)):
        (tp,tn,fp,fn)=predict(ltrain,ztrain,total,popn[sel])  # Selection of tuples
        acc=(tp+tn)/(tp+tn+fp+fn)
        acclist[sel]=acc
        totalfitness+=acc
        cumulfitness.append(totalfitness)
        if tempmax<acc:
            tempmax=acc
            tempmaxindex=sel
    for i in range(pop_size):
        k=random.random()
        for j in range(len(cumulfitness)):
            if cumulfitness[j]>k:
                temppopn[i]=popn[j]
                break
    popn=temppopn[:]
    
    ncrossover=int(pop_size*cross_rate)
    
    crosslist=random.sample(range(0, len(ltrain[0])-1), ncrossover)   # Crossover
    cpoint=random.randint(0,len(ltrain[0])-1)
    for i in range(len(crosslist)):
        if i==len(crosslist)-1:
            t1=popn[i][cpoint:]
            t2=popn[0][cpoint:]
            popn[i]=popn[i][:cpoint]+t2
            popn[0]=popn[0][:cpoint]+t1
            continue
        
        t1=popn[i][cpoint:]
        t2=popn[i+1][cpoint:]
        popn[i]=popn[i][:cpoint]+t2
        popn[i+1]=popn[i+1][:cpoint]+t1
        
    nmutation=int(mutation*pop_size*len(popn[0]))
    for i in range(nmutation):                        # Mutations
        t1=random.randint(0,pop_size-1)
        t2=random.randint(0,len(popn[0])-1)
        if popn[t1][t2]==0:
            popn[t1][t2]=1
        else:
            popn[t1][t2]=0
    if Max==tempmax:
        break
    else:
        Max=tempmax
        maxindex=tempmaxindex
    
print(popn[maxindex])
(tp,tn,fp,fn)=predict(ltest,ztest,total,popn[maxindex])
print('Genetic Algorithm + Naive Bayes:')
print('Accuracy =',(tp+tn)/(tp+tn+fp+fn))
print('precision(+) =',tp/(tp+fp))
print('recall(+) =',tp/(tp+fn))
print('precision(-) =',tn/(tn+fn))
print('recall(-) =',tn/(tn+fp))
print('\nRegular Naive Bayes:')
(tp,tn,fp,fn)=predict(ltest,ztest,total,[1]*len(popn[0]))
print('Accuracy =',(tp+tn)/(tp+tn+fp+fn))