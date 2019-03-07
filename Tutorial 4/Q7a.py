import numpy as np	
import matplotlib.pyplot as plt

np.random.seed(42)

rho0 = 1
a = 1
r_s = 1

def NFW(r):
   return rho0/((r/r_s)*(1+(r/r_s))**2)

def Hernquist(r):
   return 1/(r*(a+r)**3)

points = np.linspace(0.01,10,1000)
NFWvalues = NFW(points)
Hernquistvalues = Hernquist(points)

x = np.random.uniform(0,100,10**8)
y = np.random.uniform(0,100,10**8)

accepted_NFW = (y<=NFW(x))
accepted_Hernquist = (y<=Hernquist(x))

accepted_NFW_x = x[accepted_NFW]
accepted_NFW_y = y[accepted_NFW]

accepted_Hernquist_x = x[accepted_Hernquist]
accepted_Hernquist_y = y[accepted_Hernquist]

fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)


axs[0].plot(points, NFWvalues, label='NFW Profile', c=(1,0,0))
axs[0].scatter(accepted_NFW_x, accepted_NFW_y, s=0.1, label='NFW RNG', c=[(0,0,1)])
axs[1].plot(points, Hernquistvalues, label='Hernquist Profile', color=(0,1,0))
axs[1].scatter(accepted_Hernquist_x, accepted_Hernquist_y, s=0.1, label='Hernquist RNG', c=[(0,0,1)])

xlabel = ['x', 'x']
ylabel = ['rho', 'rho']
#ticks = [range(-100,160,20), range(0,100,10)]
i=0
for ax in axs:
    ax.set(xlabel=xlabel[i], ylabel=ylabel[i])#, xticks = ticks[i])
    i+=1

axs[0].legend()
axs[1].legend()
plt.show()
