"""

       Load and visualize a precomputed control barrier function (CBF) for the single integrator.

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
cbf_file_name = '2025-03-24_17-33-42_s2_cbfm.json'

cbf_folder_name = r'Examples_paper\S1\Data'

cbf_offset = 0

# cbf_file_name = '2025-02-11_16-27-19_myCBFmodule'
# cbf_folder_name = 'Data'

orientation_value = 0  # Orientation angle in radians (first plot)
x_value = -4.0  # Fixed x value for the slice (second plot)

color_map = 'viridis'  # Colormap for visualization

# Load the precomputed CBF
cbfModule = CBFmodule()
cbfModule.load(filename=cbf_file_name, folder_name=cbf_folder_name)
cbfModule.cbf.cbf_values += cbf_offset

X, Y = cbfModule.cbf.domain
cbf_values = cbfModule.cbf.cbf_values

# set nan values to zero
cbf_values[np.isnan(cbf_values)] = 0

# compute the h-function for each point in the domain
h = cbfModule.h
h_values = np.zeros_like(cbf_values)
for i in range(len(X)):
    for j in range(len(Y)):
        h_values[i, j] = cbfModule.h([X[i, j], Y[i, j]])

diff = h_values - cbf_values


# Enable interactive mode (only for Python scripts)
plt.ion()

# Extract the 2D slice
cbf_slice = cbf_values[:, :]
X_slice = X[:, :]
Y_slice = Y[:, :]

# Create mask for where X and Y have entries inside a specified interval
domain_to_plot = [-10, 10, -10, 10] # [x_min, x_max, y_min, y_max]
mask = (X_slice >= domain_to_plot[0]) & (X_slice <= domain_to_plot[1]) & \
       (Y_slice >= domain_to_plot[2]) & (Y_slice <= domain_to_plot[3])

# Apply mask to slice: remove points outside the interval
X_slice = np.ma.masked_array(X_slice, mask=~mask)
Y_slice = np.ma.masked_array(Y_slice, mask=~mask)
cbf_slice = np.ma.masked_array(cbf_slice, mask=~mask)
diff = np.ma.masked_array(diff, mask=~mask)

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
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')

# Plot translucent surface first
ax.plot_surface(X_slice, Y_slice, cbf_slice, cmap=color_map, alpha=1.0, edgecolor='none')

# Plot sorted scatter points second (respects depth occlusion)
ax.scatter(X_sorted, Y_sorted, cbf_sorted, c=colors_sorted, s=30, edgecolor='black', depthshade=True)

# Labels and title
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("CBF Value")
ax.set_xlim(domain_to_plot[0], domain_to_plot[1])
ax.set_ylim(domain_to_plot[2], domain_to_plot[3])

# Display the interactive plot
plt.show()

input("Press Enter to continue...")

########################################################################################
# Create 3d plot for difference between CBF and h

# Enable interactive mode (only for Python scripts)
plt.ion()

norm_diff = mcolors.Normalize(vmin=diff.min(), vmax=diff.max())
cmap_diff = matplotlib.colormaps[color_map]

# Sort points by depth (from back to front)
X_flat, Y_flat, diff_flat = X_slice.ravel(), Y_slice.ravel(), diff.ravel()
depth = X_flat + Y_flat + diff_flat  # Approximate depth for sorting

sort_idx = np.argsort(depth)  # Sort from farthest to nearest
X_sorted, Y_sorted, diff_sorted = X_flat[sort_idx], Y_flat[sort_idx], diff_flat[sort_idx]
colors_sorted = cmap_diff(norm_diff(diff_sorted))  # Get corresponding colors

# Create the 3D plot
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')

# Plot translucent surface first
ax.plot_surface(X_slice, Y_slice, diff, cmap=color_map, alpha=1.0, edgecolor='none')

# Plot sorted scatter points second (respects depth occlusion)
# ax.scatter(X_sorted, Y_sorted, diff_sorted, c=colors_sorted, s=30, edgecolor='black', depthshade=True)

# Labels and title
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("difference")
ax.set_xlim(domain_to_plot[0], domain_to_plot[1])
ax.set_ylim(domain_to_plot[2], domain_to_plot[3])

input("Press Enter to close the plots...")
plt.close('all')