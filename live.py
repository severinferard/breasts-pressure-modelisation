import serial
import time
import numpy as np
import pyvista as pv
from pyvista import examples
import pyvistaqt as pvqt
from scipy.spatial import KDTree

np.set_printoptions(linewidth=500)

N_COLUMNS = 21
N_ROWS = 12

ser = serial.Serial('/dev/tty.usbmodem21301', 115200, timeout=1)


def read_sample():
    val = ser.readline()

    while not b'\n' in val:
        val += ser.readline()

    try:
        data = np.array(val.decode().strip().split(','))
        if (len(data) != N_COLUMNS * N_ROWS):
            print("Invalid data")
            return None

    except UnicodeDecodeError:
        return None

    return data.astype(np.int32)


points = np.load("./assets/boob_grid_12_21.npy")


# Remove points skipped (at 0, 0, 0))
points_not_zero = points[np.sum(points, axis=1) != 0]
point_cloud = pv.PolyData(points)
point_cloud['pressure'] = np.full(N_ROWS * N_COLUMNS, 1)


mesh = pv.read('./assets/boob.obj')
mesh.translate(np.array([0, -0.035, 0]), inplace=True)


k = 5

kdtree = KDTree(points)

mesh['nearest_point'] = kdtree.query(mesh.points)[1]
mesh['nearest_points'] = kdtree.query(mesh.points, k=k)[1]
mesh['nearest_points_dist'] = kdtree.query(mesh.points, k=k)[0]
# mesh['pressure'] = np.full(mesh.points.shape, 1)


# data_flatten = data.flatten()

# Step 1: Calculate the reciprocal of the distances
reciprocal_distances = 1 / mesh['nearest_points_dist']

# Step 2: Square the reciprocal distances
squared_reciprocal_distances = reciprocal_distances ** 1.2

# Step 3: Normalize the weights
normalized_weights = squared_reciprocal_distances / \
    np.sum(squared_reciprocal_distances, axis=1, keepdims=True)

mesh['pressure'] = np.sum(np.full(N_ROWS * N_COLUMNS, 1)
                          [mesh['nearest_points']] * normalized_weights, axis=1)


# plotter = pvqt.BackgroundPlotter()
# plotter.add_mesh(point_cloud, scalars='pressure',  cmap='cool', point_size=15)
# plotter.add_mesh(point_cloud, scalars='pressure',  cmap='cool', point_size=15)
# plotter.view_isometric()


pl = pvqt.BackgroundPlotter(shape=(1, 2), border=False)
pl.subplot(0, 0)
pl.add_text("Measured Pressure Values on Grid", font_size=12)
pl.add_mesh(point_cloud, scalars='pressure',
            cmap='cool', point_size=15, clim=[0, 50])

pl.subplot(0, 1)
pl.add_text(f"Inverse Distance Weighted Interpolation (k={k})", font_size=12)
pl.add_mesh(mesh, scalars='pressure', cmap='cool', clim=[0, 50])

pl.link_views()

# pl = pvqt.BackgroundPlotter(shape=(1, 2), border=False)
# pl.subplot(0, 0)
# pl.add_text("Measured Pressure Values on Grid", font_size=12)


# pl.add_mesh(point_cloud, scalars='pressure', cmap='cool', point_size=15)

calibration_data = None

while True:

    val = ser.readline()

    while not b'\n' in val:
        val += ser.readline()

    data = np.array(val.decode().strip().split(','))

    if (len(data) != N_COLUMNS * N_ROWS):
        print("Invalid data")
        continue

    data = data.astype(np.int32)

    if (calibration_data is None):
        calibration_data = data

    data = np.subtract(data, calibration_data)

    # point_cloud['pressure'][:] = data
    # point_cloud.points = point_cloud.points + 0.001
    # point_cloud['pressure'][:] = np.full(N_ROWS * N_COLUMNS, 3)
    point_cloud['pressure'][:] = data
    mesh['pressure'] = np.sum(
        data[mesh['nearest_points']] * normalized_weights, axis=1)
    # mesh['pressure'] = np.full(mesh.points.shape, 25)

    print(mesh['pressure'])

    # print(point_cloud['pressure'])

    # plotter.render()
    # plotter.app.processEvents()
    # plotter.update_scalars()
    # pl.update_scalar_bar_range([0, 100])
    pl.render()
    pl.app.processEvents()
