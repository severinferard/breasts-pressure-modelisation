import vtk
import numpy as np
import noise
import serial.tools.list_ports
from scipy.spatial import KDTree
from vtkmodules.util import numpy_support


class VTKPointCloudRenderer:
    def __init__(self, serial_port,  n_columns=21, n_rows=12, point_size=30, K=5):
        self.N_COLUMNS = n_columns
        self.N_ROWS = n_rows
        self.noise_frame = 0
        self.ser = serial.Serial(serial_port, 115200, timeout=1)
        self.K = K

        self.currentData = None
        self.calibrationData = None

        self.points = vtk.vtkPoints()
        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetNumberOfComponents(3)  # RGB colors
        self.colors.SetName("Colors")

        self.meshColors = vtk.vtkUnsignedCharArray()
        self.meshColors.SetNumberOfComponents(3)  # RGB colors
        self.meshColors.SetName("Mesh Colors")

        # Load your mesh points
        boobMeshCloudPoints = np.load("./assets/boob_grid_12_21.npy")
        for p in boobMeshCloudPoints:
            self.points.InsertNextPoint(p)

        reader = vtk.vtkOBJReader()
        reader.SetFileName('./assets/boob.obj')
        reader.Update()
        self.meshPolyData = reader.GetOutput()
        self.meshPolyDataPoints = numpy_support.vtk_to_numpy(self.meshPolyData.GetPoints().GetData())

        # Translate the mesh
        transform = vtk.vtkTransform()
        transform.Translate(0, -0.035, 0)  # Adjust the translation values as needed
        transformFilter = vtk.vtkTransformFilter()
        transformFilter.SetInputData(self.meshPolyData)
        transformFilter.SetTransform(transform)
        transformFilter.Update()
        self.meshPolyData = transformFilter.GetOutput()
        self.meshPolyData.GetPointData().SetScalars(self.meshColors)

        # Create a mapper for the mesh
        self.meshMapper = vtk.vtkPolyDataMapper()
        self.meshMapper.SetInputData(self.meshPolyData)

        # Create an actor for the mesh
        self.meshActor = vtk.vtkActor()
        self.meshActor.SetMapper(self.meshMapper)

        origin = np.array([0, 0, 0])
        self.points_to_skip = np.where((boobMeshCloudPoints == origin).all(axis=1))[0]
        self.points_of_interest = np.delete(boobMeshCloudPoints, self.points_to_skip, axis=0)

        # Create the point cloud and initialize the scalars
        cloudPressure = np.zeros(self.points_of_interest.shape[0])
        cloudMaxPressure = np.zeros(self.points_of_interest.shape[0])

        # Create a kd-tree for quick nearest-neighbor lookup.
        kdtree = KDTree(self.points_of_interest)

        # Find the K nearest point_cloud points for each points in the boob mesh and calculate their respective distances
        self.meshNearestPOIs = kdtree.query(self.meshPolyDataPoints, k=self.K)[1]
        self.meshNearestPOIsDists = kdtree.query(self.meshPolyDataPoints, k=self.K)[0]

        # # Step 1: Calculate the reciprocal of the distances
        reciprocal_distances = 1 / self.meshNearestPOIsDists

        # # Step 2: Square the reciprocal distances
        squared_reciprocal_distances = reciprocal_distances ** 1.2

        # # Step 3: Normalize the weights
        self.normalizedWeight = squared_reciprocal_distances / \
            np.sum(squared_reciprocal_distances, axis=1, keepdims=True)

        meshPressure = np.zeros(self.meshPolyDataPoints.shape[0])
        meshMaxPressure = meshPressure
        # boob_mesh['pressure'] = np.zeros(boob_mesh.points.shape[0])
        # boob_mesh['max_pressure'] = boob_mesh['pressure']

        # Create a polydata object to hold the points and colors
        self.pointPolydata = vtk.vtkPolyData()
        self.pointPolydata.SetPoints(self.points)
        self.pointPolydata.GetPointData().SetScalars(self.colors)

        # Create a vertex glyph filter to render points
        self.vertexFilter = vtk.vtkVertexGlyphFilter()
        self.vertexFilter.SetInputData(self.pointPolydata)
        self.vertexFilter.Update()

        # Create a mapper
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.vertexFilter.GetOutputPort())

        # Create an actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetPointSize(point_size)

        # Create a renderer
        self.renderer = vtk.vtkRenderer()
        # self.renderer.AddActor(self.actor)
        self.renderer.AddActor(self.meshActor)

        self.renderer.SetBackground(0.1, 0.1, 0.1)  # Set background color

        # Automatically adjust the camera to view the points
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().SetPosition(0.062098271192713925, 0.3756974835244739, 0.13730874265772017)

        # Create a render window
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.AddRenderer(self.renderer)
        self.renderWindow.SetSize(self.renderWindow.GetScreenSize())

        # Create an interactor
        self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow)

        # Set up a repeating timer to call the update function
        self.timer_id = self.renderWindowInteractor.CreateRepeatingTimer(20)  # 100ms interval

        # Add an observer to the timer event to update the points and colors
        self.renderWindowInteractor.AddObserver('TimerEvent', self.update_points_colors)

    def read_noise(self):
        scale = 10.0
        octaves = 4
        persistence = 0.5
        lacunarity = 2.0
        speed = 0.1

        data = np.zeros((self.N_ROWS, self.N_COLUMNS))

        for i in range(self.N_ROWS):
            for j in range(self.N_COLUMNS):
                # Calculate Perlin noise for each point in the grid
                data[i, j] = noise.pnoise3(i / scale,
                                           j / scale,
                                           self.noise_frame * speed,
                                           octaves=octaves,
                                           persistence=persistence,
                                           lacunarity=lacunarity,
                                           repeatx=1024,
                                           repeaty=1024,
                                           base=0)
        self.noise_frame += 1
        return data.flatten() * 100

    def read_sample(self):
        val = self.ser.readline()

        while not b'\n' in val:
            val += self.ser.readline()

        try:
            data = np.array(val.decode().strip().split(','))
            if (len(data) != self.N_COLUMNS * self.N_ROWS):
                print("Invalid data")
                return None

        except UnicodeDecodeError:
            return None

        data = data.astype(np.int32)
        self.currentData = data

        if self.calibrationData is None:
            self.calibrate()
        return self.currentData - self.calibrationData

    def calibrate(self):
        self.calibrationData = self.currentData

    def update_points_colors(self, caller, event):

        values = self.read_sample()
        if values is None:
            return

        values = np.delete(values, self.points_to_skip)
        values = np.clip(values, 0, None)

        # self.colors.Reset()  # Clear previous colors
        # for noise_value in values:
        #     # Normalize noise value to the range [0, 1]
        #     normalized_value = max(0, min(noise_value, 50)) / 50

        #     # Map normalized value to color (from blue to red)
        #     r = int(255 * normalized_value)
        #     g = 0
        #     b = int(255 * (1 - normalized_value))
        #     self.colors.InsertNextTuple3(r, g, b)

        print(self.renderer.GetActiveCamera().GetPosition())
        MAX_VALUE = 30

        self.meshColors.Reset()
        pressure = np.sum(values[self.meshNearestPOIs] * self.normalizedWeight, axis=1)

        normalized_values = np.clip(pressure, 0, MAX_VALUE) / MAX_VALUE
        colors = np.column_stack(
            [normalized_values * 255, np.zeros_like(normalized_values), np.full_like(normalized_values, 255)])

        vtkData = numpy_support.numpy_to_vtk(num_array=colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

        self.meshPolyData.GetPointData().SetScalars(vtkData)
        self.meshPolyData.Modified()  # Mark the polydata as modified to update the rendering

        # Update the colors in the polydata
        # self.pointPolydata.GetPointData().SetScalars(self.colors)
        # self.pointPolydata.Modified()  # Mark the polydata as modified to update the rendering

        self.renderWindow.Render()  # Re-render the window

    def start(self):
        # Start the interaction loop
        self.renderWindowInteractor.Start()


if __name__ == "__main__":
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if port.name.startswith("cu.usbmodem"):
            selected_port = port
            break
    else:
        selected_port = None

    if not selected_port:
        print("No matching port found.")
        exit(1)

    print(f"Using port {selected_port.device}")
    renderer = VTKPointCloudRenderer(selected_port.device)
    renderer.start()
