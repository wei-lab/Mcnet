import csv
import matplotlib.pyplot as plt
import numpy as np
def getSensitivity(predict,labels):
     TP=0
     FN=0
     for i in range(len(predict)):
         if(labels[i]==1 and predict[i]==1):
             TP+=1
         elif (labels[i]==1 and predict[i]==0):
             FN+=1
     return float(TP)/(TP+FN)

def getSpecificity(predict,labels):
    TN=0;
    FP=0;
    for i in range(len(predict)):
        if(labels[i]==0 and predict[i]==0):
            TN+=1
        elif(labels[i]==0 and predict[i]==1):
            FP+=1
    return float(TN)/(TN+FP)

def getPrecision(predict,labels):
    TP=0
    FP=0
    for i in range(len(predict)):
        if(labels[i]==1 and predict[i]==1):
            TP+=1
        elif(labels[i]==0 and predict[i]==1):
            FP+=1
    return float(TP)/(TP+FP)
def getAccuracy(predict,labels):
    T=0;
    for i in range(len(predict)):
        if((labels[i]==1 and predict[i]==1) or (labels[i]==0 and predict[i]==0)):
            T+=1
    return float(T)/len(predict)

def getF1Score(precision,recall):

    return 2*(precision*recall/(precision+recall))

def getAUC(score):
    a=sorted(score,reverse=True)

    k=len(a)+1
    R=0
    m=0
    n=0
    for i in a:
        if(i[1]==0):
            R+=k;
            m+=1
        else:
            n+=1
        k-=1

    return (R-m*(m+1)/2)/float((m*n))




f=file("predictresult.csv")
#froc=open("/home/zhanghao/PycharmProjects/CNN-fish-classfication/cn/ynu/Summers/ROC.txt",'w')
reader = csv.reader(f)
output=[]
lable=[]
predict=[]
for line in reader:
    #output.append(float(line[0][1:][:-1].split()[0]))
    predict.append(int(line[0]))
    lable.append(int(line[-1]))
X=[]
Y=[]
#print "AUC:",getAUC(zip(output,lable))

print "Accuracy:",getAccuracy(predict,lable)
print "Precision:",getPrecision(predict,lable)
print "recall,sensitivity:",getSensitivity(predict,lable)
print "Specificity:", getSpecificity(predict,lable)
print 'F1-score',getF1Score(getPrecision(predict,lable),getSensitivity(predict,lable))
for i in range(0,101,1):

    temp=map(lambda x:0 if x<float(i)/100 else 1,output)
    X.append(getSensitivity(temp,lable))
    Y.append(1-getSpecificity(temp,lable))
    #froc.write(str(getSensitivity(temp,lable))+","+str(1-getSpecificity(temp,lable))+"\n")

#froc.close()
print X
print Y
plt.plot(X,Y,"-*")
plt.show()
