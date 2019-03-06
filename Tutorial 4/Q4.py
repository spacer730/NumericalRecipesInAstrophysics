import numpy as np	
from mayavi import mlab

np.random.seed(42)

def LCG(x):
   a = 1664525
   c = 1013904223
   k = 32
   m = 2**k
   return (a*x+c)%m

x=[2]

for i in range(100):
   x.append(LCG(x[-1]))

p_u1 = np.random.uniform(0,1,5000)
p_u2 = np.random.uniform(0,1,5000)
r = 1
theta1 = np.pi*p_u1
theta2 = np.arccos(1-2*p_u1)
phi = 2*np.pi*p_u2

def spheretocart(r,theta,phi):
   x = r*np.sin(theta)*np.cos(phi)
   y = r*np.sin(theta)*np.sin(phi)
   z = r*np.cos(theta)
   return x, y, z

x1, y1, z1 = spheretocart(r,theta1,phi)
x2, y2, z2 = spheretocart(r,theta2,phi)

figure = mlab.figure('myfig', bgcolor=(1, 1, 1), fgcolor = (0,0,0))
figure.scene.disable_render = True

mlab.points3d(x1, y1, z1, np.full(len(x1),1), scale_factor = 0.01, color=(1,0,0), figure = figure)
mlab.points3d(x2, y2, z2, np.full(len(x2),1), scale_factor = 0.01, color=(0,1,0), figure = figure)
mlab.axes(nb_labels = 4, figure = figure)

figure.scene.disable_render = False 
mlab.show()
