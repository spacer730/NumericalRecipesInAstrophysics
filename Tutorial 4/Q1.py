import numpy as np
import matplotlib.pyplot as plt

valuerange = np.linspace(0,2*np.pi,200)
analytical = 2*valuerange*np.sin(valuerange)+(valuerange**2)*np.cos(valuerange)
plt.plot(valuerange,analytical,c=(0,0,0))

def func(x):
	return (x**2)*(np.sin(x))

def centraldifference(x,h):
	return (func(x+h)-func(x-h))/(2*h)

h=[0.1,0.01,0.001]
labels=['0.1','0.01','0.001']
centraldif = [centraldifference(valuerange,h[i]) for i in range(len(h))]

for i in range(len(centraldif)):
	plt.plot(valuerange,centraldif[i],label=h[i])

def riddler(x,h,d,m):
	D=[[] for i in range(m)]
	D[0].append(centraldifference(x,h))
	if m>1:
		for i in range(m-1):
			D[0].append(centraldifference(x,h/(d**(i+1))))

	for j in range(m-1):
		riddlercombine(D,j,d,m)		
	return D[-1][-1]

def riddlercombine(D,j,d,m):
	for i in range(m-j-1):
		D[j+1].append((d**(2*(j+1))*D[j][i+1]-D[j][i])/(d**(2*(j+1))-1))


riddler=riddler(valuerange,0.1,2,5)
plt.plot(valuerange,riddler)
plt.legend()
plt.show()
