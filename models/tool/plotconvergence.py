import csv
import matplotlib.pyplot as plt
import numpy as np

def openfile(file):
    x=[]
    y=[]
    with open(file,'r') as f:
        for line in f.readlines():
            x.append(line.split(":")[0])
            y.append(line.split(":")[1])
    return x,y

def show_convergence_speed(x,y):
    plt.plot(x[:3000],y[:3000])
    plt.show()

x,y=openfile('../logs/loss.txt')

show_convergence_speed(x,y)