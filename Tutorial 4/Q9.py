import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def MCIntegrationCircle(N,h,k,r):
	inside=[]
	outside=[]
	x = np.random.uniform(h-r,h+r,N)
	y = np.random.uniform(k-r,k+r,N)
	radii = ((x-h)**2 + (y-k)**2)**.5
	inside_x = x[radii<=r]
	inside_y = y[radii<=r]

	outside_x = x[radii>r]
	outside_y = y[radii>r]

	estimatedarea = (len(inside_x)/(len(x)))*4*r**2
	actualarea = np.pi*(r**2)

	return estimatedarea, inside_x, inside_y, outside_x, outside_y, x, y

est, inside_x, inside_y, outside_x, outside_y, x, y = MCIntegrationCircle(1000,2,3,3)

estimatedarea = (len(inside_x)/(len(x)))*4*3**2
actualarea = np.pi*(3**2)

print("The estimated area is: " + str(estimatedarea))
print("The actual area is: " + str(actualarea))

estimatelist = []
for i in range(20000):
   estimatelist.append(MCIntegrationCircle(1000,2,3,3)[0])

fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].scatter(inside_x, inside_y, c=[(0,1,0)])
axs[0].scatter(outside_x, outside_y, c=[(1,0,0)])
axs[1].hist(estimatelist, bins=50)

xlabel = ['x', 'estimatedarea']
ylabel = ['y', 'counts']
#ticks = [range(-100,160,20), range(0,100,10)]
i=0
for ax in axs:
    ax.set(xlabel=xlabel[i], ylabel=ylabel[i])#, xticks = ticks[i])
    i+=1

#axs[0].set_xlim([-1,5])
#axs[0].set_ylim([0,6])

axs[1].axvline(x=actualarea, ymin=0, ymax=1, color=(1,0,0))

plt.show()
