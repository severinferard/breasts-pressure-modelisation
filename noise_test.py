import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import noise


# Set the dimensions of the grid
width, height = 100, 100

# Set parameters for Perlin noise
scale = 10.0
octaves = 4
persistence = 0.5
lacunarity = 2.0
speed = 0.1  # Speed of the "movement" of noise

# Initialize the grid
grid = np.zeros((height, width))

# Initialize the figure and axis
fig, ax = plt.subplots()
ax.set_axis_off()
im = ax.imshow(grid, cmap='gray', vmin=-1, vmax=1, interpolation='lanczos')

# Function to update the grid for each frame


def update(frame):
    global grid
    for i in range(height):
        for j in range(width):
            # Calculate Perlin noise for each point in the grid
            grid[i, j] = noise.pnoise3(i/scale,
                                       j/scale,
                                       frame * speed,
                                       octaves=octaves,
                                       persistence=persistence,
                                       lacunarity=lacunarity,
                                       repeatx=1024,
                                       repeaty=1024,
                                       base=0)
    im.set_array(grid)
    return [im]


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)

plt.show()
