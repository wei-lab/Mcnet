from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
f=open("/home/zhanghao/PycharmProjects/CNN-fish-classfication/cn/ynu/MutipathToRGBHSV/loss.txt")
y=[]
for line in f:
    y.append(float(line.split(':')[1][:-3]))


x=[i for i in range(1400)]

plt.plot(x,y[4600:6000])
plt.show()

f1=open("/home/zhanghao/PycharmProjects/CNN-fish-classfication/cn/ynu/lenet/loss.txt")
y1=[]
for line in f1:
    y1.append(float(line.split(':')[1][:-3]))


x1=[i for i in range(len(y1))]

plt.plot(x1,y1)
plt.show()