import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
   gal11_numberofhaloes = 20390835
   gal12_numberofhaloes = 1904612
   gal13_numberofhaloes = 21488
   gal14_numberofhaloes = 223
   gal15_numberofhaloes = 2
   
   fh = open('Data/satgals_m14.txt')
   haloes = [[] for i in range(223)]
   halo_index = -1
   for line in fh:
      line = line.rstrip('\n')
      if line == '#':
         print(line)
         halo_index += 1
         print(halo_index)
      else:
         haloes[halo_index].append(line)
   fh.close()
   
   haloes = [[[float(s) for s in haloes[i][j].split('   ')] for j in range(len(haloes[i]))] for i in range(len(haloes))]

   x = haloes(x[:,0],x[:,1],x[:,2])

   f = np.genfromtxt("Data/satgals_m14.txt")
