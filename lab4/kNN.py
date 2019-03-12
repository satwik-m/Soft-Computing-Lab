import pandas as pd
from random import randrange,shuffle

class kNN:
    def __init__(self,cls_labels,k):
        self.cls_labels=cls_labels
        self.k=k
        
    def predict(self,train_data,test_data):
        count=0
        for k in range(len(test_data)):
            row=test_data[k]
            dst=[None for i in range(len(train_data))]
            for j in range(len(train_data)):
                dst[j]=self.ecl_dst(row,train_data[j])
                
            c=[0 for i in range(len(self.cls_labels))]
            for i in range(self.k):
            	mini=min(dst)
            	index=dst.index(mini)
            	dst[index]=float("inf")
            	c[train_data[index][0]]+=1

            if self.cls_labels[test_data[k][0]]==self.cls_labels[c.index(max(c))]:
            	count+=1

        print('acc=',count*100/len(test_data),'%')
        return count*100/len(test_data)

    def ecl_dst(self,row,train_data):
    	dst=0
    	for i in range(len(train_data)):
    		dst+=(row[i]-train_data[i])**2
    	dst=dst**0.5
    	return dst

def main():
    df=pd.read_csv("SPECT.csv")
    dataset=df.values.tolist()
    shuffle(dataset)
    
    #identifying the class,attributes'/variables' labels
    cls_labels=[]
    for row in dataset:
        if row[0] not in cls_labels:
            cls_labels.append(row[0])
    
    #encoding the class labels
    for row in dataset:
        row[0]=cls_labels.index(row[0])
        
    #print(cls_labels)
    #k-fold cross validation
    n_folds=10
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    
    k=11
    knn=kNN(cls_labels,k)
    average=0
    for fold in dataset_split:
        train_data=list(dataset_split)
        train_data.remove(fold)
        train_data=sum(train_data,[])
        test_data=fold
        average+=knn.predict(train_data,test_data)#predicting on test data
    print('average accuracy =',average/10)
    
if __name__=="__main__":
    main()