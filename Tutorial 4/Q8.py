import numpy as np	
from mayavi import mlab

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

r = np.random.uniform(0,100,10**8)
rho = np.random.uniform(0,100,10**8)

accepted_Hernquist = (rho<=Hernquist(r))

accepted_Hernquist_r = r[accepted_Hernquist]
accepted_Hernquist_rho = rho[accepted_Hernquist]

p_u1 = np.random.uniform(0,1,len(accepted_Hernquist_r))
p_u2 = np.random.uniform(0,1,len(accepted_Hernquist_r))
theta = np.arccos(1-2*p_u1)
phi = 2*np.pi*p_u2

def spheretocart(r,theta,phi):
   x = r*np.sin(theta)*np.cos(phi)
   y = r*np.sin(theta)*np.sin(phi)
   z = r*np.cos(theta)
   return x, y, z

accepted_Hernquist_x, accepted_Hernquist_y, accepted_Hernquist_z = spheretocart(accepted_Hernquist_r,theta,phi)

box = ((np.abs(accepted_Hernquist_x) <=2) & (np.abs(accepted_Hernquist_y) <=2) & (np.abs(accepted_Hernquist_z) <=2))

figure = mlab.figure('myfig', bgcolor=(1, 1, 1), fgcolor = (0,0,0))
figure.scene.disable_render = True

mlab.points3d(accepted_Hernquist_x[box], accepted_Hernquist_y[box], accepted_Hernquist_z[box], np.full(len(accepted_Hernquist_x[box]),1), scale_factor = 0.01, color=(1,0,0), figure = figure)
mlab.axes(nb_labels = 4, figure = figure)

figure.scene.disable_render = False 
mlab.show()
