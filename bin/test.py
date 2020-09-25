import numpy as np
from sklearn.utils.extmath import cartesian

ldofs = [np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([0, 1, 2])] 

xdofs = [] 
ydofs = [] 
zdofs = []
for x in ldofs[0]:
    for y in ldofs[1]:
        for z in ldofs[2]:
            xdofs.append(x)
            ydofs.append(y)
            zdofs.append(z)

shape = [ idx.size for idx in ldofs ]
dim = len(shape)
gdofs = [None] * dim
gdofs[0] = ldofs[0].repeat(np.prod(shape[1:])).tolist()
for k in range(1, dim):
    gdofs[k] = np.tile(ldofs[k].repeat(np.prod(shape[k:])), np.prod(shape[0:k])).tolist()

print(xdofs, "\n",
gdofs[0], "\n",
ydofs, "\n",
gdofs[1], "\n",
zdofs, "\n",
gdofs[2])

'''
x = ldofs[0].repeat(np.prod([ idx.size for idx in ldofs[1:] ])).tolist()
y = [ np.tile(ldofs[k].repeat(np.prod([ idx.size for idx in ldofs[:-k] ])), np.prod([ idx.size for idx in ldofs[k+1:] ])).tolist() for k in range(1,len(ldofs)-1) ]
z = np.tile(ldofs[-1], np.prod([ idx.size for idx in ldofs[:-1] ])).tolist()
'''



