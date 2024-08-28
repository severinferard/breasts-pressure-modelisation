
# import vtk
# import numpy as np
# import noise


# N_COLUMNS = 21
# N_ROWS = 12


# def read_noise():
#     global noise_frame
#     scale = 10.0
#     octaves = 4
#     persistence = 0.5
#     lacunarity = 2.0
#     speed = 0.01

#     data = np.zeros((N_ROWS, N_COLUMNS))

#     for i in range(N_ROWS):
#         for j in range(N_COLUMNS):
#             # Calculate Perlin noise for each point in the grid
#             data[i, j] = noise.pnoise3(i/scale,
#                                        j/scale,
#                                        (noise_frame) * speed,
#                                        octaves=octaves,
#                                        persistence=persistence,
#                                        lacunarity=lacunarity,
#                                        repeatx=1024,
#                                        repeaty=1024,
#                                        base=0)
#     noise_frame += 1
#     return data.flatten() * 100


# # Create a point cloud of random points
# numPoints = 1000
# points = vtk.vtkPoints()
# colors = vtk.vtkUnsignedCharArray()
# colors.SetNumberOfComponents(3)  # RGB colors
# colors.SetName("Colors")

# # Center of the point cloud (assume origin here, but could be any point)
# center = np.array([0.0, 0.0, 0.0])

# boobMeshCloud = np.load("./assets/boob_grid_12_21.npy")
# for p in boobMeshCloud:
#     points.InsertNextPoint(p)


# # Create a polydata object to hold the points and colors
# pointPolydata = vtk.vtkPolyData()
# pointPolydata.SetPoints(points)
# pointPolydata.GetPointData().SetScalars(colors)

# # Create a vertex glyph filter to render points
# vertexFilter = vtk.vtkVertexGlyphFilter()
# vertexFilter.SetInputData(pointPolydata)
# vertexFilter.Update()

# # Create a mapper
# mapper = vtk.vtkPolyDataMapper()
# mapper.SetInputConnection(vertexFilter.GetOutputPort())

# # Create an actor
# actor = vtk.vtkActor()
# actor.SetMapper(mapper)
# actor.GetProperty().SetPointSize(5)  # Increase point size

# # Create a renderer
# renderer = vtk.vtkRenderer()
# renderer.AddActor(actor)
# renderer.SetBackground(0.1, 0.1, 0.1)  # Set background color

# # Automatically adjust the camera to view the points
# renderer.ResetCamera()

# # Create a render window
# renderWindow = vtk.vtkRenderWindow()
# renderWindow.AddRenderer(renderer)
# renderWindow.SetSize(800, 600)

# # Create an interactor
# renderWindowInteractor = vtk.vtkRenderWindowInteractor()
# renderWindowInteractor.SetRenderWindow(renderWindow)

# # Start the visualization
# renderWindow.Render()
# renderWindowInteractor.Start()


import vtk
import numpy as np
import noise

N_COLUMNS = 21
N_ROWS = 12
noise_frame = 0  # Initialize the noise frame counter


def read_noise():
    global noise_frame
    scale = 10.0
    octaves = 4
    persistence = 0.5
    lacunarity = 2.0
    speed = 0.1

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


# Create a point cloud of points loaded from the file
points = vtk.vtkPoints()
colors = vtk.vtkUnsignedCharArray()
colors.SetNumberOfComponents(3)  # RGB colors
colors.SetName("Colors")

# Load your mesh points
boobMeshCloud = np.load("./assets/boob_grid_12_21.npy")
for p in boobMeshCloud:
    points.InsertNextPoint(p)

# Create a polydata object to hold the points and colors
pointPolydata = vtk.vtkPolyData()
pointPolydata.SetPoints(points)
pointPolydata.GetPointData().SetScalars(colors)

# Create a vertex glyph filter to render points
vertexFilter = vtk.vtkVertexGlyphFilter()
vertexFilter.SetInputData(pointPolydata)
vertexFilter.Update()

# Create a mapper
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(vertexFilter.GetOutputPort())

# Create an actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetPointSize(30)  # Increase point size

# Create a renderer
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.1, 0.1)  # Set background color

# Automatically adjust the camera to view the points
renderer.ResetCamera()

# Create a render window
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindow.SetSize(renderWindow.GetScreenSize())

# Create an interactor
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)


def update_points_colors(caller, event):
    # Generate noise-based color data
    noise_values = read_noise()

    colors.Reset()  # Clear previous colors
    for noise_value in noise_values:
        # Normalize noise value to the range [0, 1]
        normalized_value = (noise_value + 100) / 200.0

        # Map normalized value to color (from blue to red)
        r = int(255 * normalized_value)
        g = 0
        b = int(255 * (1 - normalized_value))
        colors.InsertNextTuple3(r, g, b)

    # Update the colors in the polydata
    pointPolydata.GetPointData().SetScalars(colors)
    pointPolydata.Modified()  # Mark the polydata as modified to update the rendering

    renderWindow.Render()  # Re-render the window


# Set up a repeating timer to call the update function
timer_id = renderWindowInteractor.CreateRepeatingTimer(20)  # 100ms interval

# Add an observer to the timer event to update the points and colors
renderWindowInteractor.AddObserver('TimerEvent', update_points_colors)

# Start the interaction loop
renderWindowInteractor.Start()
