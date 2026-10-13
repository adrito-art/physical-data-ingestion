import pandas as pd
import numpy as np

df = pd.read_parquet("merged_sample.parquet")
clean_episodes = []

for ep_idx, group in df.groupby('episode_index'):
    # 1. Duration Check: Drop episodes shorter than 4 seconds
    if (group['timestamp'].max() - group['timestamp'].min()) < 4.0:
        continue
        
    # 2. Kinematic Extraction
    states = np.vstack(group['observation.state'].values)
    dt = group['timestamp'].diff().bfill().values.reshape(-1, 1)
    dq = np.diff(states, axis=0, prepend=states[0:1])
    
    # Calculate velocity (handling division by zero safely if dt is 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        vel = np.abs(dq / dt)
        vel = np.nan_to_num(vel, posinf=0.0) # Convert infinities to 0
    velocity = pd.DataFrame(vel)
    
    # 3. Micro-Glitch vs Real Spike Detection
    # Flag if ANY joint exceeds 15 rad/s for 4 consecutive timestamps
    is_spike = velocity > 15.0
    sustained_spikes = is_spike.rolling(window=4, min_periods=4).sum() == 4
    
    if sustained_spikes.any().any():
        print(f"Dropped Episode {ep_idx}: Hardware spike detected.")
        continue
        
    clean_episodes.append(group)

if clean_episodes:
    clean_df = pd.concat(clean_episodes, ignore_index=True)
    print(f"QA Complete: Retained {clean_df['episode_index'].nunique()} clean episodes.")
else:
    print("QA Complete: 0 episodes retained. All were flagged as anomalous or too short.")