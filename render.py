import serial
import time
import numpy as np
import pyvista as pv
from pyvista import examples
import pyvistaqt as pvqt

np.set_printoptions(linewidth=500)

N_COLUMNS=21
N_ROWS=12

# table = np.reshape(array, (N_COLUMNS, N_ROWS)).astype(int)

x = np.arange(0, N_ROWS, 1.0)
y = np.arange(0, N_COLUMNS, 1.0)

xx, yy = np.meshgrid(x, y)
z = np.full((N_COLUMNS, N_ROWS), 0)

grid = pv.StructuredGrid(xx, yy, z)
# grid.plot()



plotter = pvqt.BackgroundPlotter()
plotter.add_mesh(grid, cmap='bwr', show_scalar_bar=False)
# plane = pv.Plane(i_size=N_COLUMNS, j_size=N_ROWS)
# plane['values'] = np.full((N_COLUMNS, N_ROWS), 0)
# plotter.add_mesh(plane, cmap='bwr')

plotter.view_isometric()



# while True:
#     plotter.render()
#     plotter.app.processEvents()

while True:                             
    val = ser.readline()               

    while not b'\n' in val:
        val += ser.readline()
    
    table = get_2d_array_from_raw_data(val)
    if (table is None):
        print("bad data")
        continue

    if calibration_table is None:
        calibration_table = table
    table = np.subtract(table, calibration_table)

    table = np.square(table)

    table = np.where(table < 100, 0, table)
    table = np.interp(table, [0, 1000], [0, -1])

    g = pv.StructuredGrid(xx, yy, table)
    grid.shallow_copy(g)
    
    plotter.render()
    plotter.app.processEvents()

    print(table)
    print('\n\n')
