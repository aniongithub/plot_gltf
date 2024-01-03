from pygltflib import GLTF2
from gltf_utils import get_animation_data, skeleton_visitor_BFS, Visitors
import matplotlib.pyplot as plt

# Define the file paths for the gltf file
TPOSE_FILENAME = "data/tpose.glb"

# Load and extract animation data from the tpose gltf object
tpose_gltf = GLTF2.load(TPOSE_FILENAME)
tpose_data = get_animation_data(tpose_gltf)

# Create a matplotlib figure and axes for plotting
fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection='3d')

# Plot the bind pose skeleton using BFS and our visitor
# Ignore the elongated feet, it's a result of automatic axis-scaling in matplotlib
skeleton_visitor_BFS(tpose_gltf, Visitors.plot_bindpose, 
                     axes = ax)
plt.savefig("bindpose.png") # save

# Clear the axes
ax.clear()

# Plot the pose at time 0
skeleton_visitor_BFS(tpose_gltf, Visitors.plot_pose_at, 
                     time = 0, animation = tpose_data, axes = ax)
plt.savefig("pose_at_t0.png") # save