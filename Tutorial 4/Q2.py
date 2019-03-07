import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

a = 1664525
c = 1013904223
k = 32
m = 2**k

def LCG(x,a,c,m):
   return (a*x+c)%m

x=[2]

for i in range(100):
   x.append(LCG(x[-1],a,c,m))

n = np.array(x[:-1])
np1 = np.array(x[1:])

def PCC(x,y):
   return (np.mean(x*y)-np.mean(x)*np.mean(y))/((np.var(x,ddof=1)*np.var(y,ddof=1))**0.5)

coefLCG = PCC(n,np1)

print("Pearson Correlation Coefficient for LCG with a="+str(a)+",c="+str(c)+"and m=2^"+str(k)+" is:"+str(coefLCG))

numpyrandomlist = np.random.uniform(0,m,100)
numpyn = numpyrandomlist[:-1]
numpynp1 = numpyrandomlist[1:]

coefNUMPY = PCC(numpyn,numpynp1)
print("Pearson Correlation Coefficient for numpy.random.uniform(0,2^"+str(k)+",100) is:"+str(coefNUMPY))

fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].scatter(n, np1, marker="o", color=(1,0,0))
axs[1].scatter(numpyn, numpynp1, marker="o", color=(0,1,0))

axs[0].text(0,4.2*10**9,'PCC: '+str(coefLCG))
axs[1].text(0,4.2*10**9,'PCC: '+str(coefNUMPY))

xlabel = ['LCG n', 'numpy n']
ylabel = ['LCG n+1', 'numpy n+1']
#ticks = [range(-100,160,20), range(0,100,10)]
i=0
for ax in axs:
    ax.set(xlabel=xlabel[i], ylabel=ylabel[i])#, xticks = ticks[i])
    i+=1

#axs[0].set_xlim([-1,5])
#axs[0].set_ylim([0,6])

#plt.show()
plt.savefig('Q2-x0=2-a='+str(a)+'-c='+str(c)+'-m='+str(m))
