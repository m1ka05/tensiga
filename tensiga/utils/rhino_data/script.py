import numpy as np
import pyvista as pv

inner = np.loadtxt('cps_inner.txt')
outer = np.loadtxt('cps_outer.txt')

# clean up machine precision
inner[inner < 1e-13] = 0
outer[outer < 1e-13] = 0

npts = inner.shape[0]
inner = inner[:,0:3]/inner[:,3].reshape(npts, -1)
outer = outer[:,0:3]/outer[:,3].reshape(npts, -1)


pc_inner = pv.PolyData(inner)
pc_outer = pv.PolyData(outer)

pc_inner.plot(eye_dome_lighting=True)
pc_outer.plot(eye_dome_lighting=True)

plotter = pv.Plotter()
plotter.add_mesh(pc_inner, color='blue', point_size=10.,
                 render_points_as_spheres=True)
plotter.add_mesh(pc_outer, color='red', point_size=10.,
                 render_points_as_spheres=True)
plotter.show_grid()
plotter.show()

