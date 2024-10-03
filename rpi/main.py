import pyvista as pv
from pyvista import examples
from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
import asyncio
import random
import numpy as np
import os
from scipy.spatial import KDTree
import noise
import serial
import serial.tools.list_ports
# Always set PyVista to plot off screen with Trame
pv.OFF_SCREEN = True

# server = get_server()
# state, ctrl = server.state, server.controller

# mesh = examples.load_random_hills()

# pl = pv.Plotter()
# pl.add_mesh(mesh)


MAX_VALUE = 150
MIN_VALUE = 30

N_COLUMNS = 21
N_ROWS = 12
K = 5


noise_frame = 0


def read_sample(ser):
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


def read_noise():
    global noise_frame
    scale = 10.0
    octaves = 4
    persistence = 0.5
    lacunarity = 2.0
    speed = 0.01

    data = np.zeros((N_ROWS, N_COLUMNS))

    for i in range(N_ROWS):
        for j in range(N_COLUMNS):
            # Calculate Perlin noise for each point in the grid
            data[i, j] = noise.pnoise3(i/scale,
                                       j/scale,
                                       (noise_frame) * speed,
                                       octaves=octaves,
                                       persistence=persistence,
                                       lacunarity=lacunarity,
                                       repeatx=1024,
                                       repeaty=1024,
                                       base=0)
    noise_frame += 1

    # Normalize the data to be between 0 and 1
    data = (data - data.min()) / (data.max() - data.min())

    # Scale the data to be between MIN_VALUE and MAX_VALUE
    data = data * (MAX_VALUE - MIN_VALUE) + MIN_VALUE

    return data.flatten()


async def main():
    server = get_server()
    state, ctrl = server.state, server.controller

    # ports = serial.tools.list_ports.comports()
    # for port in ports:
    #     if port.name.startswith("cu.usbmodem"):
    #         selected_port = port
    #         break
    # else:
    #     selected_port = None

    # if not selected_port:
    #     print("No matching port found.")
    #     exit(1)

    # print(f"Using port {selected_port.device}")

    # ser = serial.Serial(selected_port.device, 115200, timeout=1)

    dir = os.path.dirname(__file__)
    assets_dir = os.path.join(dir, './assets/')
    points = np.load(os.path.join(assets_dir, 'boob_grid_12_21.npy'))

    # # Remove points that are outside of the boob mesh.
    # # Because those points have been skipped during the points mapping, their value is (0, 0, 0)
    origin = np.array([0, 0, 0])
    points_to_skip = np.where((points == origin).all(axis=1))[0]
    points_of_interest = np.delete(points, points_to_skip, axis=0)

    # # Create the point cloud and initialize the scalars
    point_cloud = pv.PolyData(points_of_interest)
    point_cloud['pressure'] = np.zeros(points_of_interest.shape[0])
    point_cloud['max_pressure'] = np.zeros(points_of_interest.shape[0])

    # # Load the boob 3D model from file
    boob_mesh = pv.read(os.path.join(assets_dir, 'boob.obj'))
    boob_mesh.translate(np.array([0, -0.035, 0]), inplace=True)

    # # Create a kd-tree for quick nearest-neighbor lookup.
    kdtree = KDTree(points_of_interest)

    # Find the K nearest point_cloud points for each points in the boob mesh and calculate their respective distances
    boob_mesh['nearest_points'] = kdtree.query(boob_mesh.points, k=K)[1]
    boob_mesh['nearest_points_dist'] = kdtree.query(boob_mesh.points, k=K)[0]

    # Step 1: Calculate the reciprocal of the distances
    reciprocal_distances = 1 / boob_mesh['nearest_points_dist']

    # Step 2: Square the reciprocal distances
    squared_reciprocal_distances = reciprocal_distances ** 1.2

    # Step 3: Normalize the weights
    normalized_weights = squared_reciprocal_distances / \
        np.sum(squared_reciprocal_distances, axis=1, keepdims=True)

    boob_mesh['pressure'] = np.zeros(boob_mesh.points.shape[0])
    boob_mesh['max_pressure'] = boob_mesh['pressure']

    pl = pv.Plotter(shape=(1, 2), border=False)
    pl.subplot(0, 0)
    pl.add_text("Measured Pressure Values on Grid", font_size=12)
    pl.add_mesh(point_cloud, scalars='pressure', cmap='cool',
                point_size=5, clim=[MIN_VALUE, MAX_VALUE])

    pl.subplot(0, 1)
    pl.add_text(f"Inverse Distance Weighted Interpolation (k={K})", font_size=12)
    pl.add_mesh(boob_mesh, scalars='pressure', cmap='cool', clim=[MIN_VALUE, MAX_VALUE])

    pl.link_views()

    pl.camera_position = 'xy'
    pl.camera.elevation += 100

    with SinglePageLayout(server) as layout:
        layout.title.set_text("Breasties 2.0")
        with layout.content:
            # Use PyVista's Trame UI helper method
            #  this will add UI controls
            view = plotter_ui(pl, add_menu=False)

    async def loop():
        calibration_data = None
        while True:
            # point_cloud['pressure'] += 1
            # data = read_sample(ser)
            data = read_noise()

            if (data is None):
                continue

            # if (calibration_data is None):
            #     calibration_data = data

            # data = np.subtract(data, calibration_data)
            data = np.delete(data, points_to_skip)

            # print(data[0])

            point_cloud['pressure'] = data
            point_cloud['max_pressure'] = np.maximum(point_cloud['max_pressure'], data)

            boob_mesh['pressure'] = np.sum(data[boob_mesh['nearest_points']] * normalized_weights, axis=1)
            pl.render()
            await asyncio.sleep(0.030)

    await asyncio.gather(loop(), server.start(exec_mode="coroutine", open_browser=False, host="0.0.0.0"))

asyncio.run(main())
