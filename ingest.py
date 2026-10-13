import pandas as pd
from pathlib import Path
from huggingface_hub import snapshot_download

# Scope Control: Download only ONE dataset chunk directory
repo_id = "lerobot/aloha_sim_insertion_scripted"
local_dir = "./sample_data"
snapshot_download(repo_id=repo_id, repo_type="dataset", 
                  allow_patterns=["data/chunk-000/*"], local_dir=local_dir)

# The PyArrow Fix: Load chunked Parquet files natively
parquet_files = list(Path(local_dir).glob("data/chunk-000/*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

df.to_parquet("merged_sample.parquet")
print(f"Successfully loaded {len(df)} frames.")