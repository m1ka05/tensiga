import numpy as np
import tensiga.utils.varmeshes as varmeshes
import pyvista as pv

# init input geometry
domain = varmeshes.Halfpipe3D(10, 8, 15)
nodes = [np.linspace(0,1,5)[1:-1] ] * 3
idomain = varmeshes.interpolation_mesh(domain, nodes)

# eval geometry
ep = [ np.linspace(0,1,100) ] * idomain.dim
ep = [ np.unique(kv) for kv in idomain.kv ]
x, y, z = [ idomain.eval(ep, k) for k in range(idomain.dim) ]

# plot at evaluation points
plotter = pv.Plotter()
mesh = pv.StructuredGrid(x, y, z)
plotter.add_mesh(mesh, show_edges=True)
plotter.show_axes()
plotter.view_xy()
plotter.show()
