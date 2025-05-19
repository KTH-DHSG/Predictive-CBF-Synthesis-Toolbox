"""

       Load and visualize a precomputed control barrier function (CBF) for the bicycle model.

       (c) Adrian Wiltz, 2025

"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from CBF.CBFmodule import CBFmodule
import matplotlib.pyplot as plt

# Parameters
cbf_file_name = '2025-03-22_00-17-20_b1_1_cbfm_2p8.json'
cbf_folder_name = r'Examples_paper\B1\Data'

cbf_offset = -2

# cbf_file_name = '2025-02-11_16-27-19_myCBFmodule'
# cbf_folder_name = 'Data'

orientation_value = 0  # Orientation angle in radians (first plot)
x_value = -4.0  # Fixed x value for the slice (second plot)

color_map = 'viridis'  # Colormap for visualization

# Load the precomputed CBF
cbfModule = CBFmodule()
cbfModule.load(filename=cbf_file_name, folder_name=cbf_folder_name)
cbfModule.cbf.cbf_values += cbf_offset

X, Y, PSI = cbfModule.cbf.domain
cbf_values = cbfModule.cbf.cbf_values

# Preprocessing of CBF data for visualization
orientation_index = np.argmin(np.abs(PSI[0,0,:] - orientation_value))  # Find closest index
orientation_actual_value = PSI[0,0,orientation_index]  # Actual orientation value
print(f"Plot CBF for the fixed orientation {orientation_actual_value} rad")

# Enable interactive mode (only for Python scripts)
plt.ion()

# Extract the 2D slice
cbf_slice = cbf_values[:, :, orientation_index]
X_slice = X[:, :, orientation_index]
Y_slice = Y[:, :, orientation_index]

# Create mask for where X and Y have entries inside a specified interval
domain_to_plot = [-10, 10, -10, 10] # [x_min, x_max, y_min, y_max]
mask = (X_slice >= domain_to_plot[0]) & (X_slice <= domain_to_plot[1]) & \
       (Y_slice >= domain_to_plot[2]) & (Y_slice <= domain_to_plot[3])

# Apply mask to slice: remove points outside the interval
X_slice = np.ma.masked_array(X_slice, mask=~mask)
Y_slice = np.ma.masked_array(Y_slice, mask=~mask)
cbf_slice = np.ma.masked_array(cbf_slice, mask=~mask)

# Normalize function values for colormap
norm = mcolors.Normalize(vmin=cbf_slice.min(), vmax=cbf_slice.max())
cmap = matplotlib.colormaps[color_map]

# Sort points by depth (from back to front)
X_flat, Y_flat, cbf_flat = X_slice.ravel(), Y_slice.ravel(), cbf_slice.ravel()
depth = X_flat + Y_flat + cbf_flat  # Approximate depth for sorting
sort_idx = np.argsort(depth)  # Sort from farthest to nearest
X_sorted, Y_sorted, cbf_sorted = X_flat[sort_idx], Y_flat[sort_idx], cbf_flat[sort_idx]
colors_sorted = cmap(norm(cbf_sorted))  # Get corresponding colors

# Create the 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot translucent surface first
ax.plot_surface(X_slice, Y_slice, cbf_slice, cmap=color_map, alpha=1.0, edgecolor='none')

# Plot sorted scatter points second (respects depth occlusion)
ax.scatter(X_sorted, Y_sorted, cbf_sorted, c=cbf_sorted, cmap=color_map, s=30, edgecolor='black', depthshade=True)

# Labels and title
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("CBF Value")
ax.set_xlim(domain_to_plot[0], domain_to_plot[1])
ax.set_ylim(domain_to_plot[2], domain_to_plot[3])
ax.set_title(f"CBF Slice at Orientation = {orientation_actual_value} rad")

# Display the interactive plot
plt.show()

input("Press Enter to continue...")

########################################################################################

# Determine the zero super level set (points where the value is close to zero)

x_fine = np.linspace(np.nanmin(X), np.nanmax(X), 500)
y_fine = np.linspace(np.nanmin(Y), np.nanmax(Y), 500)
psi_fine = np.linspace(np.nanmin(PSI), np.nanmax(PSI), 500)

interpolator = cbfModule.cbf.getCbfInterpolator(method='linear')

X_fine, Y_fine, PSI_fine = np.meshgrid(x_fine, y_fine, psi_fine)

cbf_values_fine = interpolator((X_fine, Y_fine, PSI_fine))

# Determine the zero-level set
epsilon = 0.1  # Threshold to approximate the zero-level set
mask = np.abs(cbf_values_fine) < epsilon  # Find points where the function is close to zero

# Get the coordinates where the condition is met
x_points = X_fine[mask]
y_points = Y_fine[mask]
psi_points = PSI_fine[mask]

num_level_lines = 12
level_lines = np.linspace(np.nanmin(PSI), np.nanmax(PSI), num_level_lines)

cmap = matplotlib.colormaps.get_cmap('viridis')
norm = matplotlib.colors.Normalize(vmin=np.nanmin(PSI), vmax=np.nanmax(PSI))
black_rgba = mcolors.to_rgba('black')

colors = [black_rgba if np.any(np.isclose(psi_point, level_lines, atol=4e-2)) else cmap(norm(psi_point)) for psi_point in psi_points]

# Prepare surface plot for the zero-level set
# Create meshgrid for interpolation
# grid_x, grid_y = np.meshgrid(np.linspace(min(x_points), max(x_points), 100),
#                              np.linspace(min(y_points), max(y_points), 100))

# Interpolate psi values on the grid
# grid_psi = griddata((x_points, y_points), psi_points, (grid_x, grid_y), method='linear')  # 'cubic' interpolation

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the interpolated surface
# ax.plot_surface(grid_x, grid_y, grid_psi, cmap='viridis')

# Plot the scattered points (zero-level set points)
ax.scatter(x_points, y_points, psi_points, c=colors, s=5, label='Zero-level set points')

ax.set_zlim([-np.pi, np.pi])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('$\psi$')
# ax.set_title('Zero-level Set')

plt.show()

########################################################################################

input("Press Enter to close plots...")