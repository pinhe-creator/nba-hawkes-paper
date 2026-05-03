"""
Step A: Download full-league NBA PBP data for 2022-23, 2023-24, 2024-25.

Source: shufinskiy/nba_data (GitHub raw)
Files:  nbastats_2022.tar.xz  (2022-23 season — starts in 2022)
        nbastats_2023.tar.xz  (2023-24 season)
        nbastats_2024.tar.xz  (2024-25 season — already have, will skip if exists)

After download, this script:
  1. Extracts each tar.xz
  2. Loads each CSV
  3. Concatenates into one big DataFrame
  4. Saves as full_league_3_seasons.csv (~50-100 MB)

NOTE: If your disk is tight, you can delete the .tar.xz and .csv per-season files
      after the combined CSV is saved.
"""

import urllib.request
import tarfile
from pathlib import Path
import pandas as pd
import os
import sys
import time

BASE_URL = "https://github.com/shufinskiy/nba_data/raw/main/datasets/{name}.tar.xz"
SEASONS = ["nbastats_2022", "nbastats_2023", "nbastats_2024"]
SEASON_LABELS = {"nbastats_2022": "2022-23", "nbastats_2023": "2023-24", "nbastats_2024": "2024-25"}

# ---------- Step 1: Download ----------
for name in SEASONS:
    tar_file = f"{name}.tar.xz"
    csv_file = f"{name}.csv"
    if Path(csv_file).exists():
        print(f"✓ {csv_file} already exists, skipping")
        continue
    if not Path(tar_file).exists():
        url = BASE_URL.format(name=name)
        print(f"\nDownloading {url} ...")
        t0 = time.time()
        urllib.request.urlretrieve(url, tar_file)
        size_mb = Path(tar_file).stat().st_size / 1e6
        print(f"  ✓ {size_mb:.1f} MB in {time.time()-t0:.1f}s")
    else:
        print(f"✓ {tar_file} already downloaded, skipping download")
    # Extract
    print(f"Extracting {tar_file} ...")
    with tarfile.open(tar_file, 'r:xz') as tar:
        tar.extractall(path='.')
    print(f"  ✓ Extracted {csv_file}")

# ---------- Step 2: Load and concatenate ----------
print("\n" + "=" * 60)
print("Loading and concatenating CSVs...")
print("=" * 60)

dfs = []
for name in SEASONS:
    csv_file = f"{name}.csv"
    label = SEASON_LABELS[name]
    print(f"\nLoading {csv_file} (season {label})...")
    df = pd.read_csv(csv_file, low_memory=False)
    df["SEASON"] = label
    print(f"  Rows: {len(df):,}  |  Games: {df['GAME_ID'].nunique()}")
    dfs.append(df)

full = pd.concat(dfs, ignore_index=True)
print(f"\n[Combined] {len(full):,} rows  |  {full['GAME_ID'].nunique()} unique games  |  3 seasons")

# ---------- Step 3: Save ----------
output_csv = "full_league_3_seasons.csv"
print(f"\nSaving {output_csv} ...")
full.to_csv(output_csv, index=False)
size_mb = Path(output_csv).stat().st_size / 1e6
print(f"  ✓ {size_mb:.1f} MB")

# ---------- Step 4: Quick validation ----------
print("\n[Sanity check]")
games_per_season = full.groupby("SEASON")["GAME_ID"].nunique()
print(games_per_season)
sub_count = (full["EVENTMSGTYPE"] == 8).sum()
to_count = (full["EVENTMSGTYPE"] == 9).sum()
print(f"\nTotal substitution events:  {sub_count:,}")
print(f"Total timeout events:       {to_count:,}")

print("\n" + "=" * 60)
print("Step A complete.")
print("=" * 60)
print("\nNext: upload full_league_3_seasons.csv to Claude.")
print("File should be ~150-300 MB. If too large for upload, we can stay with")
print("the LAL+DAL data — Claude will guide you if compression is needed.")