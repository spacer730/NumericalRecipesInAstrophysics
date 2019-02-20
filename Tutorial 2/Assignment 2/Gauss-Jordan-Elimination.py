import numpy as np
import matplotlib.pyplot as plt

A0=np.array([[3,8,1,-12,-4],[1,0,0,-1,0],[4,4,3,-40,-3],[0,2,1,-3,-2],[0,1,0,-12,0]])
A0=A0.astype(float)
b0=np.array([2,0,1,0,0])
b0=b0.astype(float)

def GaussJordan(A,b):
    for i in range(np.shape(A)[1]):
        j=i
        pivotfound=False
        while ((j<(np.shape(A)[1]))&(pivotfound==False)):
            if (A[j][i]!=0):
                pivotfound=True
                pivot = A[j][i]
                print("The pivot is")
                print(pivot)
            else:
                j+=1
        if j!=i:
            A[[j,i]]=A[[i,j]] #Swapping rows j and i
            b[[j,i]]=b[[i,j]] #Swapping rows j and i
            print(A)
            print(b)
        A[i]=A[i]/pivot
        b[i]=b[i]/pivot
        print(A)
        print(b)
        reducerows = list(range(np.shape(A)[1]))
        reducerows.remove(j)
        for k in reducerows:
            A[k]=A[k]-A[k][i]*A[i]
            b[k]=b[k]-b[k]*A[k][i]
        print(A)
        print(b)
    return A, b

A1, b1 = GaussJordan(A0,b0)
