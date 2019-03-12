import pandas as pd
import numpy as np
import random
import math

class iNode:
    def __init__(self):
        self.input=0
        self.output=None
        self.weights=None
        
class oNode:
    def __init__(self):
        self.input=0
        self.output=None
        self.bias=None
        self.error=None
        
class innerNode:
    def __init__(self):
        self.input=0
        self.output=None
        self.bias=None
        self.weights=None
        self.error=None
        
class MLP:
    def __init__(self,inodes,n,innerlayers,bias,weight1,weight2,learning_rate):
        self.rate=learning_rate
        self.outputnode=oNode()
        self.layers=[[innerNode() for i in range(n)] for j in range(innerlayers)]
        self.layers=[[iNode() for i in range(inodes)]]+self.layers+[[self.outputnode]]
        for layer in range(len(self.layers)):
            if layer==0:
                for node in range(len(self.layers[layer])):
                    self.layers[layer][node].weights=[weight1 for i in range(len(self.layers[layer+1]))]
            
            elif layer==len(self.layers)-1:
                for node in range(len(self.layers[layer])):
                    #self.layers[layer][node].weights=[weight2 for i in range(len(self.layers[layer+1]))]
                    self.layers[layer][node].bias=bias
                    
            else:
                for node in range(len(self.layers[layer])):
                    self.layers[layer][node].weights=[random.uniform(0,1) for i in range(len(self.layers[layer+1]))]
                    self.layers[layer][node].bias=bias
                    
    def fun(self,x):
        return 1/(1+math.exp(-1*x))
    
    def derivative(self,x):
        return (x.output*(1-x.output))
    
    def classification(self,x):
        if x>0.5:
            return 1
        return 0
                    
    def train(self,ltrain,ztrain):
        number=0
        for input in ltrain:
            for node in range(len(input)):
                self.layers[0][node].output=input[node]
            
            for layer in range(1,len(self.layers)):           #forward propagation
                for node in range(len(self.layers[layer])):
                    self.layers[layer][node].input=self.layers[layer][node].bias
                    for i in self.layers[layer-1]:
                        self.layers[layer][node].input+=(i.weights[node]*i.output)
                    self.layers[layer][node].output=self.fun(self.layers[layer][node].input)
            self.outputnode.output=self.classification(self.outputnode.output)
            
            self.outputnode.error=(ztrain[number]-self.outputnode.output)   #back propagation
            for layer in range(len(self.layers)-2,-1,-1):
                for node in range(len(self.layers[layer])):
                    self.layers[layer][node].error=0
                    for j in range(len(self.layers[layer+1])):
                        self.layers[layer][node].error+=(self.layers[layer][node].weights[j]*self.layers[layer+1][j].error)
                    self.layers[layer][node].error*=self.derivative(self.layers[layer][node])
                                                                    
            for i in range(0,len(self.layers)-1):
                for j in range(len(self.layers[i])):
                    for k in range(len(self.layers[i][j].weights)):
                        self.layers[i][j].weights[k]+=self.rate*self.layers[i+1][k].error*self.layers[i][j].output
                        
                    if i!=0:
                        self.layers[i][j].bias+=(self.rate*self.layers[i][j].error)
            number+=1
            
    def test(self,ltest,ztest):
        number=0
        count=0
        tp=0
        tn=0
        fp=0
        fn=0
        for input in ltest:
            for node in range(len(input)):
                self.layers[0][node].output=input[node]
            
            for layer in range(1,len(self.layers)):           #forward propagation
                for node in range(len(self.layers[layer])):
                    self.layers[layer][node].input=self.layers[layer][node].bias
                    for i in self.layers[layer-1]:
                        self.layers[layer][node].input+=(i.weights[node]*i.output)
                    self.layers[layer][node].output=self.fun(self.layers[layer][node].input)
            self.outputnode.output=self.classification(self.outputnode.output)
            #print(self.outputnode.output)
            if self.outputnode.output==ztest[number]:
                count+=1
                if ztest[number]==1:
                    tp+=1
                else:
                    tn+=1
            elif ztest[number]==1:
                fn+=1
            else:
                fp+=1
            number+=1
        print('accuracy =',count/len(ztest)*100,'%')
        print('precision(+) =',tp/(tp+fp))
        print('recall(+) =',tp/(tp+fn))
        print('precision(-) =',tn/(tn+fn))
        print('recall(-) =',tn/(tn+fp))
        


data=pd.read_csv('SPECT.csv')
copy=data.iloc[:,:]
ltrain=[]
ztrain=[]
ltest=[]
ztest=[]
dict={}
c=0
n=data.iloc[:,1].count()

for i in range(n):            # dividing train and test dataset
    row=random.randint(0,copy.iloc[:,1].count()-1)
    if i<0.9*n:
        ltrain.append(copy.iloc[row,1:])
        ztrain.append(copy.iloc[row,0])
        if copy.iloc[row,0] not in dict:
            dict[copy.iloc[row,0]]=c
            c+=1
        copy.drop(copy.index[row])
    else:
        ltest.append(copy.iloc[row,1:])
        ztest.append(copy.iloc[row,0])
        copy.drop(copy.index[row])
        
for i in range(len(ztrain)): #changing categorical variables to numbers
    ztrain[i]=dict[ztrain[i]]
for i in range(len(ztest)):
    ztest[i]=dict[ztest[i]]  
inputs=len(ltrain[0])
M=MLP(inputs,5,1,1/6,1/(5*inputs),1/5,0.2)
M.train(ltrain,ztrain)
M.test(ltest,ztest)
        