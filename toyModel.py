#come up with a vector field v(t,x) = [1, x]
#instantiate a bunch of points at random (uniform random on [-5,5] x [-5,5])
import numpy as np
import random

pts = 10 * (np.random.rand(50, 2) - 0.5)
vels = np.ones([50,2])
vels[:,1] = pts[:,1]