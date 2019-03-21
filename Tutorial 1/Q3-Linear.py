import numpy as np
import matplotlib.pyplot as plt

numbers = np.linspace(1,20,7)

values = 2*numbers*np.sin(numbers)

def Linearinterpolation(interpolatedrange,numbers,values):
   interpolatedvalues = []

   linslope = []

   for i in range(len(numbers)-1):
      linslope.append((values[i+1]-values[i])/(numbers[i+1]-numbers[i]))

   def linearinterpolation(x,i):
      if i<6:
         return (linslope[i]*(x-numbers[i])+values[i])
      else:
         return (linslope[5]*(x-numbers[5])+values[5])

   for interpolatednumber in interpolatedrange:
      rangefound = False
      indexrange = 0
      while rangefound!=True:
         if interpolatednumber >= numbers[-1]:
            indexrange = len(numbers)-2
            rangefound = True
            interpolatedvalues.append(linearinterpolation(interpolatednumber,indexrange))
         elif numbers[indexrange] <= interpolatednumber < numbers[indexrange+1]:
            rangefound = True
            interpolatedvalues.append(linearinterpolation(interpolatednumber,indexrange))
         else:
            indexrange+=1

   return interpolatedvalues

interpolatedrange = np.linspace(1,25,100)
interpolatedvalues = Linearinterpolation(interpolatedrange,numbers,values)

plt.scatter(numbers,values)
plt.scatter(interpolatedrange,interpolatedvalues)
plt.show()
