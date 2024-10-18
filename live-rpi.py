import serial.tools.list_ports
import time
import numpy as np
import pyvista as pv
from pyvista import examples
import pyvistaqt as pvqt
from scipy.spatial import KDTree
import noise
import os
from typing import Optional, Tuple

from qtpy import QtCore
from qtpy.QtCore import Signal
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMainWindow, QWidget


class MainWindow(QMainWindow):
    """Convenience MainWindow that manages the application."""

    signal_close = Signal()
    signal_gesture = Signal(QtCore.QEvent)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        title: Optional[str] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Initialize the main window."""
        QMainWindow.__init__(self, parent=parent)
        if title is not None:
            self.setWindowTitle(title)
        if size is not None:
            self.resize(*size)
        self.setWindowFlag(Qt.FramelessWindowHint)

    def event(self, event: QtCore.QEvent) -> bool:
        """Manage window events and filter the gesture event."""
        if event.type() == QtCore.QEvent.Gesture:  # pragma: no cover
            self.signal_gesture.emit(event)
            return True
        return super().event(event)

    def closeEvent(self, event: QtCore.QEvent) -> None:  # pylint: disable=invalid-name
        """Manage the close event."""
        self.signal_close.emit()
        event.accept()


np.set_printoptions(linewidth=500)

N_COLUMNS = 21
N_ROWS = 12
K = 5

MOUNT_DIR = "/media/ubuntu/"

ports = serial.tools.list_ports.comports()

for port in ports:
    if port.name.startswith("ttyACM"):
        selected_port = port
        break
else:
    selected_port = None

if not selected_port:
    print("No matching port found.")
    exit(1)

print(f"Using port {selected_port.device}")

noise_frame = 0


def read_noise(ser):
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
    return data.flatten() * 100

# def remove_noise(data):
#     table = np.reshape(data, (N_ROWS, N_COLUMNS))
#     threshold = 10  # Define your threshold value here

#     for i in range(N_ROWS):
#         for j in range(N_COLUMNS):
#             if table[i, j] > threshold:
#                 neighbors_x = []
#                 neighbors_y = []

#                 if i > 0:
#                     neighbors_y.append(table[i-1, j])
#                 if i < N_ROWS - 1:
#                     neighbors_y.append(table[i+1, j])
#                 if j > 0:
#                     neighbors_x.append(table[i, j-1])
#                 if j < N_COLUMNS - 1:
#                     neighbors_x.append(table[i, j+1])

#                 neighbors = neighbors_x + neighbors_y
#                 print(neighbors)
#                 if any(n > 10 for n in neighbors):
#                     pass
#                 else:
#                     print("removed noise")
#                     table[i, j] = 0

#                 # if all(x <= threshold for x in neighbors_x) or all(y <= threshold for y in neighbors_y):
#                 #     print("noise ignored")
#                 #     table[i, j] = 0  # Remove the noise by setting it to 0

#                 # if not neighbors_x or not neighbors_y:
#                 #     table[i, j] = 0  # Remove the noise by setting it to 0

#     return table.flatten()


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


def get_storage_file():

    if not os.path.exists(MOUNT_DIR) or not os.path.isdir(MOUNT_DIR):
        return None

    storage_dirs = os.listdir(MOUNT_DIR)
    if (len(storage_dirs) == 0):
        return None

    storage_dir = storage_dirs[0]

    return os.path.join(MOUNT_DIR, storage_dir, time.strftime("data_%Y%m%d_%H%M%S.bin"))


dirname = os.path.dirname(__file__)

points = np.load(os.path.join(dirname, "./assets/boob_grid_12_21.npy"))

# Remove points that are outside of the boob mesh.
# Because those points have been skipped during the points mapping, their value is (0, 0, 0)
origin = np.array([0, 0, 0])
points_to_skip = np.where((points == origin).all(axis=1))[0]
points_of_interest = np.delete(points, points_to_skip, axis=0)

# Create the point cloud and initialize the scalars
point_cloud = pv.PolyData(points_of_interest)
point_cloud['pressure'] = np.zeros(points_of_interest.shape[0])
point_cloud['max_pressure'] = np.zeros(points_of_interest.shape[0])


# Load the boob 3D model from file
boob_mesh = pv.read(os.path.join(dirname, './assets/boob.obj'))
boob_mesh.translate(np.array([0, -0.035, 0]), inplace=True)

# Create a kd-tree for quick nearest-neighbor lookup.
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


MAX_VALUE = 150
MIN_VALUE = 30


pl = pvqt.BackgroundPlotter(shape=(1, 2), border=False, toolbar=False,
                            menu_bar=False, editor=False, app_window_class=MainWindow)
pl.subplot(0, 0)
pl.add_text("Measured Pressure Values on Grid", font_size=12)
pl.add_mesh(point_cloud, scalars='pressure', cmap='cool', point_size=10, clim=[MIN_VALUE, MAX_VALUE])

pl.subplot(0, 1)
pl.add_text(f"Inverse Distance Weighted Interpolation (k={K})", font_size=12)
pl.add_mesh(boob_mesh, scalars='pressure', cmap='cool', clim=[MIN_VALUE, MAX_VALUE])


# pl.subplot(1, 0)
# pl.add_text(f"Max Measured Pressure Values On Grid", font_size=12)
# pl.add_mesh(point_cloud, scalars='max_pressure', cmap='cool',
#             point_size=15, copy_mesh=True, clim=[MIN_VALUE, MAX_VALUE])

# pl.subplot(1, 1)
# pl.add_text(f"Max Inverse Distance Weighted Interpolation (k={K})", font_size=12)
# pl.add_mesh(boob_mesh, scalars='max_pressure', cmap='cool',
#             copy_mesh=True, clim=[MIN_VALUE, MAX_VALUE])

pl.link_views()

pl.camera_position = 'xy'
pl.camera.elevation += 100

calibration_data = None
previous_data = None
last_touch_time = time.time()


print("Starting loop")
storage_file_name = get_storage_file() or '/dev/null'
print(f"Storage file: {storage_file_name}")


with open(storage_file_name, "wb") as f:
    with serial.Serial(selected_port.device, 115200, timeout=1) as ser:
        while True:

            try:
                raw_data = read_sample()
            except KeyboardInterrupt:
                exit(0)
            except Exception as e:
                print(e)
                continue
            # data = read_noise()
            if (raw_data is None):
                continue

            if (calibration_data is None):
                calibration_data = raw_data

            # raw_data = remove_noise(raw_data)
            data = np.subtract(raw_data, calibration_data)
            data = np.delete(data, points_to_skip)

            f.write(data)

            point_cloud['pressure'][:] = data
            point_cloud['max_pressure'][:] = np.maximum(point_cloud['max_pressure'], data)

            boob_mesh['pressure'][:] = np.sum(data[boob_mesh['nearest_points']] * normalized_weights, axis=1)
            # boob_mesh['max_pressure'][:] = np.sum(
            #     point_cloud['max_pressure'][boob_mesh['nearest_points']] * normalized_weights, axis=1)

            pl.render()
            pl.app.processEvents()

            if previous_data is not None:
                diff = np.abs(np.subtract(data, previous_data))
                change = np.max(diff)
                if change > 100:
                    last_touch_time = time.time()
            if time.time() - last_touch_time > 10:
                print("calibrating")
                calibration_data = raw_data
                last_touch_time = time.time()
            previous_data = data
