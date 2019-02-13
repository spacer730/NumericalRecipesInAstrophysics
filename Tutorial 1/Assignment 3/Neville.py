import numpy as np
import matplotlib.pyplot as plt

numbers = np.linspace(1,20,7)
values = 2*numbers*np.sin(numbers)
plt.scatter(numbers,values,s=50,c=(0,1,0))

interpolatedvalues = []
interpolatedrange = np.linspace(1,21,100)

def nevi(x,i,j):
    return ((x-numbers[j])*p[i][j+1]-(x-numbers[j+1+i])*p[i][j])/(numbers[j+1+i]-numbers[j])    

for interpolatednumber in interpolatedrange:
    M=len(numbers)
    p=[[] for i in range(7)]
    p[0]=values
    while M>1:
        for j in range(M-1):
            order = 1+len(numbers)-M
            p[order].append(nevi(interpolatednumber,order-1,j))
        M-=1
    interpolatedvalues.append(p[len(numbers)-1][0])

plt.scatter(interpolatedrange,interpolatedvalues,s=25,c=(1,1,0))
plt.show()
