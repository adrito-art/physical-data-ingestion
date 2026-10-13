import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

# Load the data you just processed
df = pd.read_parquet("merged_sample.parquet")

# Isolate a single episode to align
ep_idx = df['episode_index'].unique()[0]
episode = df[df['episode_index'] == ep_idx].copy()

# 1. Extract original high-frequency timestamps and states
original_timestamps = episode['timestamp'].values
joint_states = np.vstack(episode['observation.state'].values)

# 2. Define the target 30Hz camera timestamps
# (Generating a timestamp every ~0.033 seconds)
camera_timestamps = np.arange(original_timestamps[0], original_timestamps[-1], 1/30.0)

# 3. The Math Engine: Cubic Spline Interpolation
# This builds a continuous mathematical curve of the robot's movement
spline = CubicSpline(original_timestamps, joint_states, axis=0)

# 4. Extract the aligned states at the exact camera timestamps
aligned_states = spline(camera_timestamps)

print(f"Episode {ep_idx} Alignment:")
print(f"Original High-Frequency States: {len(original_timestamps)} frames")
print(f"Target 30Hz Camera States: {len(camera_timestamps)} frames")
print(f"Sample Aligned State Output Shape: {aligned_states.shape}")
print("Interpolation successful. Vision and kinematics are now mathematically synchronized.")