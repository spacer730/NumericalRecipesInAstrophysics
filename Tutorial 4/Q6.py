import numpy as np	
import matplotlib.pyplot as plt

np.random.seed(42)

mu = 0
b = 1

points = np.linspace(-5,5,100)
laplace = np.exp(-np.abs(points-mu)/b)/(2*b)

y = np.random.uniform(0,1,100)
y = y[np.argsort(y)]
cdf1 = (0.5*np.exp((y[y<=0.5]-mu)/b))
cdf2 = (1-0.5*np.exp((mu-y[y>0.5])/b))
x1 = mu + b*np.log(2*y[y<=0.5])
x2 = mu - b*np.log(2*(1-y[y>0.5]))
x = np.concatenate((x1,x2))

plt.hist(x,color=(0,0,1),density=1,label='Transformation Method')
plt.plot(points,laplace,c=(0,1,0),label='Laplace Distribution')
plt.legend()
plt.show()
