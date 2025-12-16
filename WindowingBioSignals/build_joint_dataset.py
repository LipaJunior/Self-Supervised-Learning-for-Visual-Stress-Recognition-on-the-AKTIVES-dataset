from pathlib import Path
import pandas as pd

# =======================
# SETTINGS
# =======================
INPUT_ROOT = Path("WindowingBioSignals/data_label_aligned_30s")
OUTPUT_CSV = Path("joint_dataset.csv")

ACC_FILE = "ACC_features.csv"
BVP_FILE = "BVP_features.csv"
EDA_FILE = "EDA_features.csv"

# =======================
# HELPERS
# =======================

def load_and_prefix(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # keep label + time columns unprefixed
    keep = {"label_time_sec", "stress_label", "reaction_label"}
    new_cols = {}

    for c in df.columns:
        if c in keep:
            continue
        new_cols[c] = f"{prefix}{c}"

    return df.rename(columns=new_cols)


# =======================
# MAIN
# =======================

rows = []

for group_dir in INPUT_ROOT.iterdir():
    if not group_dir.is_dir():
        continue

    group = group_dir.name

    for child_dir in group_dir.iterdir():
        if not child_dir.is_dir():
            continue

        child = child_dir.name

        for game_dir in child_dir.iterdir():
            if not game_dir.is_dir():
                continue

            game = game_dir.name

            acc_path = game_dir / ACC_FILE
            bvp_path = game_dir / BVP_FILE
            eda_path = game_dir / EDA_FILE

            # Require at least ONE modality
            if not any(p.exists() for p in [acc_path, bvp_path, eda_path]):
                continue

            dfs = []

            if acc_path.exists():
                dfs.append(load_and_prefix(acc_path, "acc_"))
            if bvp_path.exists():
                dfs.append(load_and_prefix(bvp_path, "bvp_"))
            if eda_path.exists():
                dfs.append(load_and_prefix(eda_path, "eda_"))

            # Merge all on label_time_sec
            df_merged = dfs[0]
            for d in dfs[1:]:
                df_merged = df_merged.merge(
                    d,
                    on=["label_time_sec", "stress_label", "reaction_label"],
                    how="inner"
                )

            if len(df_merged) == 0:
                continue

            # Add metadata
            df_merged.insert(0, "group", group)
            df_merged.insert(1, "child", child)
            df_merged.insert(2, "game", game)

            rows.append(df_merged)

            print(f"[OK] {group}/{child}/{game} → rows: {len(df_merged)}")

# Concatenate all sessions
if not rows:
    raise RuntimeError("No data found to merge!")

joint = pd.concat(rows, ignore_index=True)

# Save
joint.to_csv(OUTPUT_CSV, index=False)

print("\nDone.")
print("Final dataset shape:", joint.shape)
print("Saved to:", OUTPUT_CSV.resolve())
