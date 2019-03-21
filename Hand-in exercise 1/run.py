import numpy as np
import matplotlib.pyplot as plt

def factorial(k):
   h = 1
   while k>1:
      h = h*k
      k -= 1
   return h

def Poisson(labda, k):
   return (labda**k)*np.exp(-labda)/(factorial(k))

def LCG(x):
   a = 3202034522624059733
   c = 4354685564936845319
   m = 2**64
   return (a*x+c)%m

def XOR_shift(x):
   a1, a2, a3 = np.uint64(21), np.uint64(35), np.uint64(4)
   x = np.uint64(x)
   x = x ^ (x >> a1)
   x = x ^ (x << a2)
   x = x ^ (x >> a3)
   return x

def RNG(length, norm = True):
   global seed

   randomnumbers = []
   state = seed
   
   for i in range(length):
      state = LCG(state)
      randomnumbers.append(XOR_shift(state))

   randomnumbers = np.array(randomnumbers)

   if norm == True:
      randomnumbers = np.array(randomnumbers)/(2**64)

   seed = state
   if length == 1:
      return randomnumbers[0]
   else:
      return randomnumbers.tolist()

def densityprofileint(x):
   return ((x/b)**(a-3))*np.exp(-(x/b)**c)*x**2

def densityprofile(x):
   return ((x/b)**(a-3))*np.exp(-(x/b)**c)

def ndprofile(x, N_sat=1):
   return N_sat*4*np.pi*A*densityprofileint(x)

def extmidpoint(func, edges, n):
   h = (edges[1]-edges[0])/n
   integration = 0

   for i in range(n):
       integration += func(edges[0]+(i+0.5)*h)
   integration = h*integration

   return integration

def extmidpointromberg(func, edges, n, N):
   s = [[] for i in range(N)]
   s[0].append(extmidpoint(func, edges, n))

   for i in range (1,N):
      n = 2*n
      s[0].append(extmidpoint(func, edges, n))
   
   for j in range(N-1):
      for i in range(N-(j+1)):
         s[j+1].append(s[j][i+1]+(s[j][i+1]-s[j][i])/(-1+4**(j+1)))

   return s[-1][0]

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

def nevi(x,i,j,numbers,p):
   return ((x-numbers[j])*p[i][j+1]-(x-numbers[j+1+i])*p[i][j])/(numbers[j+1+i]-numbers[j])

def Nevillesinterpolation(interpolationrange,numbers,values):
   interpolatedvalues = []

   for interpolatednumber in interpolatedrange:
      M=len(numbers)
      p=[[] for i in range(7)]
      p[0]=values
      while M>1:
         for j in range(M-1):
            order = 1+len(numbers)-M
            p[order].append(nevi(interpolatednumber,order-1,j,numbers,p))
         M-=1
      interpolatedvalues.append(p[len(numbers)-1][0])

   return interpolatedvalues

def centraldifference(func,x,h):
   return (func(x+h)-func(x-h))/(2*h)

def riddler(func,x,h,d,m):
   D=[[] for i in range(m)]
   D[0].append(centraldifference(func,x,h))
   if m>1:
      for i in range(m-1):
         D[0].append(centraldifference(func,x,h/(d**(i+1))))
   for j in range(m-1):
      riddlercombine(D,j,d,m)
   return D[-1][-1]

def riddlercombine(D,j,d,m):
   for i in range(m-j-1):
      D[j+1].append((d**(2*(j+1))*D[j][i+1]-D[j][i])/(d**(2*(j+1))-1))

def analyticaldrvdensityprofile(x):
   return densityprofile(x)*((1/x)*(a-3-c*(x/b)**c))

"""
def ceil(x):
   if x%1 == 0:
      return int(x)
   else:
      return int(x)+1

def floor(x):
   return int(x)
"""

def argsort(x):
   if type(x) is np.ndarray:
      xd = x[:].tolist() #Create a copy of the array so the actual array doesn't get sorted
   else:
      xd = x[:]
   y = [i for i in range(len(x))]
   argsortinner(xd,y)
   return y

def argsortinner(xd, y, start=0, end=None): #When sorting the array, also keeps track of how the indices swap around
   if end == None:
      end = len(xd)-1
   if start < end:
      index = argpivotsort(xd,y,start,end)
      argsortinner(xd,y,start,index-1)
      argsortinner(xd,y,index+1,end)

def argpivotsort(xd,y,start,end):
   pivot = xd[end]
   i = start-1
   for j in range(start,end):
      if xd[j] <= pivot:
         i += 1
         xd[i], xd[j] = xd[j], xd[i]
         y[i], y[j] = y[j], y[i]
   xd[i+1], xd[end] = xd[end], xd[i+1]
   y[i+1], y[end] = y[end], y[i+1]
   return i+1

def pivotsort(x,start,end):
   pivot = x[end]
   i = start-1
   for j in range(start,end):
      if x[j] <= pivot:
         i += 1
         x[i], x[j] = x[j], x[i]
   x[i+1], x[end] = x[end], x[i+1]
   return i+1

def Quicksort(x, start=0, end=None):
   if end == None:
      end = len(x)-1
   if start < end:
      index = pivotsort(x,start,end)
      Quicksort(x,start,index-1)
      Quicksort(x,index+1,end)

if __name__ == '__main__':
   seed = 2
   print("The seed is: " + str(seed))

   print(Poisson(1,0))
   print(Poisson(5,10))
   print(Poisson(3,20))
   print(Poisson(2.6,40))

   RNG_list = RNG(1000)
   RNG_list2 = RNG(10**6)

   n_RNG_list = np.array(RNG_list[:-1])
   np1_RNG_list = np.array(RNG_list[1:])

   fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

   axs[0].scatter(n_RNG_list, np1_RNG_list, marker="o", color=(1,0,0), facecolors='none')
   axs[1].hist(RNG_list2, bins = 20, range = (0,1))

   xlabel = ['Combined RNG n', 'Random number generated by combined RNG']
   ylabel = ['Combined RNG n+1', 'Counts']

   i=0
   for ax in axs:
      ax.set(xlabel=xlabel[i], ylabel=ylabel[i])
      i+=1

   fig.savefig('RNG-test-results')

   a = (RNG(1)*1.4)+1.1
   b = (RNG(1)*1.5)+0.5
   c = (RNG(1)*2.5)+1.5
   
   integration = extmidpointromberg(densityprofileint, [0,5], 10**2, 4)
   A = (1/(4*np.pi))*(1/integration)
   
   print("a,b,c,A = " + str(a) + "," + str(b) + "," + str(c) + "," + str(A))

   numbers = [10**-4, 10**-2, 10**-1, 1, 5]
   densityvalues = [densityprofile(10**-4), densityprofile(10**-2), densityprofile(10**-1), densityprofile(1), densityprofile(5)]
   interpolatedrange = np.logspace(-4,0.69897,100)

   linearvalues = Linearinterpolation(interpolatedrange,numbers,densityvalues)
   #Nevillesvalues = Nevillesinterpolation(interpolatedrange,numbers,densityvalues)
   
   fig2, axs2 = plt.subplots()
   axs2.loglog(interpolatedrange, linearvalues)
   axs2.set(xlabel='log(x)', ylabel='Density profile')
   fig2.savefig('Log-Log plot Linear interpolation')

   derivative_at_b = riddler(densityprofile,b,0.1,2,6)
   analyticaldrv_at_b = analyticaldrvdensityprofile(b)
   
   print("The analytical derivative at b is: " + str(analyticaldrv_at_b))
   print("The numerically solved derivative at b is: " + str(derivative_at_b))

   p_u1 = np.array(RNG(100))
   p_u2 = np.array(RNG(100))

   theta = np.arccos(1-2*p_u1)
   phi = 2*np.pi*p_u2

   x_accepted_densityprofile = []
   while len(x_accepted_densityprofile)<10000:
      x = RNG(1)*5
      y = RNG(1)*1.33
      if y <= ndprofile(x):
         x_accepted_densityprofile.append(x)

   logbins = np.logspace(np.log10(10**-4),np.log10(5),20)

   fig3, axs3 = plt.subplots()
   axs3.hist(x_accepted_densityprofile,bins=logbins,density=True, log=True)
   Quicksort(x_accepted_densityprofile)
   axs3.plot(x_accepted_densityprofile, ndprofile(np.array(x_accepted_densityprofile)))
   axs3.set_xscale('log')
   fig3.savefig('Density profile')
   
   haloes = [[] for i in range(1000)]
   
   #1000 haloes with 100 satellites each.
   for i in range(1000):
      x_local_accepted_densityprofile = []
      while len(x_local_accepted_densityprofile)<100:
         x = RNG(1)*5
         y = RNG(1)*1.33
         if y <= ndprofile(x):
            x_local_accepted_densityprofile.append(x)
      
      haloes[i] = x_local_accepted_densityprofile
   
   """
   To make histogram of the average of super_x, just concatenate all the data to one array of super_x and make a histogram and divide the histogram counts by 1000. This gives
   the average in each bin. The problem now is we want 100 values, however each x has different lengths.
   """

   flattened_haloes = [item for sublist in haloes for item in sublist]
   Quicksort(flattened_haloes)
   
   fig4, axs4 = plt.subplots()

   logbins = np.logspace(np.log10(10**-4),np.log10(5),20)

   axs4.hist(flattened_haloes,bins=logbins,weights=[1/(1000*(logbins[1]-logbins[0])) for i in range(len(flattened_haloes))], log=True)
   axs4.plot(flattened_haloes, ndprofile(np.array(flattened_haloes),N_sat=100))
   axs4.set_xscale('log')
   fig4.savefig('Density profile Haloes')

   """
   fig4, axs4 = plt.subplots()

   logbins = np.logspace(np.log10(10**-4),np.log10(5),20)

   hist, bins = np.histogram(flattened_haloes,bins=logbins)
   center = (bins[:-1]+bins[1:])/2
   width = 1*(bins[1]-bins[0])
   hist = hist/(1000*width)

   axs4.bar(center, hist, align = 'center', width = width)
   axs4.plot(flattened_haloes, ndprofile(np.array(flattened_haloes),N_sat=100))
   axs4.set_xscale('log')
   fig4.savefig('Density profile Haloes')
   """
   

"""
# Create the histogram and normalize the counts to 1
hist, bins = np.histogram(x, bins = 50)
max_val = max(hist)
hist = [ float(n)/max_val for n in hist]

# Plot the resulting histogram
center = (bins[:-1]+bins[1:])/2
width = 0.7*(bins[1]-bins[0])
plt.bar(center, hist, align = 'center', width = width)
plt.show()
"""

