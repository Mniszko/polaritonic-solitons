import numpy as np
import matplotlib.pyplot as plt

# Assuming your data is in a variable called 'solution'
# solution.shape = (time_size, space_size)

# Trim the spatial dimension (1D space)
space_start = 10  # example start index
space_end = 50    # example end index
trimmed_solution = solution[:, space_start:space_end]

# Create the plot
plt.figure(figsize=(10, 6))

# Create a binned plot (heatmap-style)
plt.imshow(trimmed_solution.T,  # Transpose to have time on x-axis
           aspect='auto',       # Adjust aspect ratio automatically
           origin='lower',      # Put t=0 at the bottom
           cmap='viridis',      # Choose a colormap
           extent=[0, trimmed_solution.shape[0],  # Time extent
                   space_start, space_end])       # Space extent

# Add labels
plt.xlabel('Time')
plt.ylabel('Space Position')
plt.colorbar(label='Solution Value')
plt.title('ODE Solution Heatmap')

plt.tight_layout()
plt.show()