import numpy as np

np.random.seed(84)

def factorial(k):
   h = 1
   while k>1:
      h = h*k
      k -= 1
   return h

def Poisson(labda, k):
   return (labda**k)*np.exp(-labda)/(factorial(k))

print(Poisson(1,0))
print(Poisson(5,10))
print(Poisson(3,20))
print(Poisson(2.6,40))

def LCG(x):
   a = 1664525
   c = 1013904223
   k = 32
   m = 2**k
   return ((a*x+c)%m)/m

def RNG():
   return 3
   #Combine LCG, 64-bit-xor-shift
