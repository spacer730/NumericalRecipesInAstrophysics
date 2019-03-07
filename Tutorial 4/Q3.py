import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

a = 237
c = 2
k = 8
m = 2**k

def ICG(x,a,c,m):
   if x!=0:
      return (a*(x**-1)+c)%m
   else:
      return c

x=[2]

for i in range(100):
   x.append(ICG(x[-1],a,c,m))

n = np.array(x[:-1])
np1 = np.array(x[1:])

def PCC(x,y):
   return (np.mean(x*y)-np.mean(x)*np.mean(y))/((np.var(x)*np.var(y))**0.5)

coefICG = PCC(n,np1)

print("Pearson Correlation Coefficient for ICG with a="+str(a)+",c="+str(c)+"and m="+str(m)+" is:"+str(coefICG))

numpyrandomlist = np.random.uniform(0,m,100)
numpyn = numpyrandomlist[:-1]
numpynp1 = numpyrandomlist[1:]

coefNUMPY = PCC(numpyn,numpynp1)
print("Pearson Correlation Coefficient for numpy.random.uniform(0,"+str(m)+",100) is:"+str(coefNUMPY))

fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].scatter(n, np1, marker="o", color=(1,0,0))
axs[1].scatter(numpyn, numpynp1, marker="o", color=(0,1,0))

axs[0].text(5,32,'PCC: '+str(coefICG))
axs[1].text(0,255,'PCC: '+str(coefNUMPY))

xlabel = ['ICG n', 'numpy n']
ylabel = ['ICG n+1', 'numpy n+1']
#ticks = [range(-100,160,20), range(0,100,10)]
i=0
for ax in axs:
    ax.set(xlabel=xlabel[i], ylabel=ylabel[i])#, xticks = ticks[i])
    i+=1

#axs[0].set_xlim([-1,5])
#axs[0].set_ylim([0,6])

plt.show()
#plt.savefig('Q3-x0=2-a='+str(a)+'-c='+str(c)+'-m='+str(m))
