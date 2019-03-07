import numpy as np	
import matplotlib.pyplot as plt

np.random.seed(42)

x0 = 0
gamma = 0.5

points = np.linspace(-30,10,100)
cauchy = 1/(np.pi*gamma*(1+((points-x0)/gamma)**2))

y = np.random.uniform(0,1,100)
y = y[np.argsort(y)]
x = x0 + gamma*np.tan(np.pi*(y-0.5))

plt.hist(x,100,color=(0,0,1),label='Transformation Method',density=1)
plt.plot(points,cauchy,c=(0,1,0),label='Cauchy Distribution')
plt.legend()
plt.show()
