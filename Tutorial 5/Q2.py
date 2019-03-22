import numpy as np

np.random.seed(42)

def bisection(func, interval): 
   return 

def secant(func, interval, criterium, maxiter):
   x_0 = interval[0]
   x_1 = interval[1]
   if func(x_0)*func(x_1)>=0:
      return None
   for i in range(maxiter):
      x_2 = (1+func(x_0)/(func(x_1)-func(x_0)))*x_1-(func(x_0)/(func(x_1)-func(x_0)))*x_0
      if abs(func(x_2)) <= criterium:
         return x_2      
      elif func(x_0)*func(x_2)<0:
         x_0 = x_0
         x_1 = x_2
      elif func(x_1)*func(x_2)<0:
         x_0 = x_1
         x_1 = x_2
      else:
         return None
   return None

def falseposition():
   return

def Brents(func, interval):
   return

def NewtonRaphson(func, drv, start, criterium):
   x_0 = start
   converged = False
   while converged == False:
      x_1 = x_0-func(x_0)/drv(x_0)
      x_0 = x_1
      if func(x_1) <= criterium:
         converged = True
   return x_1

def f_a(x):
   return x**3-6*x**2+11*x-6

def f_a_drv(x):
   return 3*x**2-12*x+11

def f_b(x):
   return np.tan(np.pi*x)-6

def f_b_drv(x):
   return np.pi/((np.cos(np.pi*x))**2)

def f_c(x):
   return x**3-2*x+2

def f_c_drv(x):
   return 3*x**2-2

def f_d(x):
   return np.exp(10*(x-1))-0.1

def f_d_drv(x):
   return 10*np.exp(10*(x-1))
