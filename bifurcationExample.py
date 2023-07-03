import random
import math
import matplotlib.pyplot as plt
import numpy as np

def genPoint(epsilon):
    t = 10*random.uniform(0,1)
    x0 = 0
    y0 = 1 + epsilon * random.uniform(0,1)

    x = t + x0
    y = (y0-1) * math.exp(0.1*t**2+x0*t) + 1
    v = (1,0.2*x*(y-1))

    return (np.array([t,x,y]), np.array(v))

numPts = 10
ptsArray = np.zeros([numPts,3])
vecArray = np.zeros([numPts,2])
for i in range(numPts):
    (ptsArray[i,:],vecArray[i,:]) = genPoint(0.1)


(T,X,Y) = (ptsArray[:,0],ptsArray[:,1],ptsArray[:,2])
(Vx,Vy) = (vecArray[:,0],vecArray[:,1])
plt.scatter(X,Y,c=T)

plt.quiver(X,Y,Vx,Vy,T)
plt.savefig('bifurcation.png')
