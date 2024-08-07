from scipy.spatial.distance import cdist
import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import random
from threading import Thread

# point cloud of random points
n_points = 100
points = np.random.random((n_points, 3))
points -= points.mean(axis=0)

# random velocities
vel = np.random.random((n_points, 3))
vel -= vel.mean(axis=0)
vel /= 1000

point_cloud = pv.PolyData(points)
point_cloud['point_color'] = np.linalg.norm(point_cloud.points, axis=1)

plotter = pvqt.BackgroundPlotter()
plotter.add_mesh(point_cloud, cmap='bwr', show_scalar_bar=False)
plotter.view_isometric()


def grav_like_force(mesh):
    """Given a point_cloud, determine simulated gravitational force."""
    dist = cdist(mesh.points, mesh.points)

    # ignore self distance
    di = np.diag_indices(dist.shape[0])
    dist[di] = 1E30

    # abritary grav constant
    grav = 1E-3

    # gravity force assuming all "objects" are of the same mass and we're just
    # using inverse distance here
    grav_f = grav*dist**-1

    # We now need to compute the direction of the force.  I'm being lazy here
    # because matrix math is hard.
    grav_f_vec = np.empty((mesh.n_points, 3))
    for ii, point in enumerate(mesh.points):
        vdist = mesh.points - point
        vdist[ii][:] = 1  # ignore self
        vdist /= np.linalg.norm(vdist, axis=1).reshape(-1, 1)
        vdist[ii][:] = 0

        grav_f_vec[ii] = grav_f[ii] @ vdist

    return grav_f_vec


while True:
    # update velocity
    vel += grav_like_force(point_cloud)
    point_cloud.points = point_cloud.points + vel*1E-3

    point_cloud['point_color'][:] = np.linalg.norm(point_cloud.points, axis=1)
    plotter.render()
    plotter.app.processEvents()
    print("something")