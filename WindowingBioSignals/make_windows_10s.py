from __future__ import annotations

from pathlib import Path
import sys
import csv
import numpy as np
import pandas as pd


# =======================
# SETTINGS
# =======================
WINDOW_SEC = 10.0
OUTPUT_DIRNAME = "data_windowed_10s_features"
EXT = ".csv"

# ignore facial data completely + files you said you don't care about
IGNORE_IF_PATH_CONTAINS = ["facial"]
IGNORE_FILENAMES = {"expertlabels.csv", "facial.csv"}  # mp4 excluded by EXT anyway


# --- make Windows console robust to UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def is_ignored(path: Path) -> bool:
    low = str(path).lower()
    if any(key in low for key in IGNORE_IF_PATH_CONTAINS):
        return True
    if path.name.lower() in IGNORE_FILENAMES:
        return True
    return False


def safe_print(*args):
    try:
        print(*args)
    except UnicodeEncodeError:
        msg = " ".join(str(a) for a in args).encode("utf-8", "replace").decode("utf-8", "replace")
        print(msg)


def find_in_root(project_root: Path) -> Path:
    """
    Find the DATA folder anywhere inside the project.
    If multiple DATA folders exist, choose the one with the most CSV files.
    """
    candidates = []

    for p in project_root.rglob("DATA"):
        if p.is_dir():
            csv_count = 0
            for f in p.rglob("*.csv"):
                if not is_ignored(f):
                    csv_count += 1
                    if csv_count >= 200:
                        break
            candidates.append((csv_count, p))

    if not candidates:
        raise FileNotFoundError(f"DATA folder not found inside project: {project_root}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def parse_time_sec_sub_str(t: str) -> float:
    """
    Parse time formatted as 'seconds,subseconds' where subseconds are fractional digits.
    Example: '1641384895,84641' -> 1641384895.084641
    """
    t = str(t).replace('"', '').strip()
    if "," not in t:
        # fallback (shouldn't happen, but don't crash the whole file)
        sec = pd.to_numeric(t, errors="coerce")
        return float(sec) if pd.notna(sec) else np.nan

    sec_str, sub_str = t.split(",", 1)
    sec = pd.to_numeric(sec_str, errors="coerce")
    if pd.isna(sec):
        return np.nan

    sub_str = (sub_str or "0").strip()
    # keep only digits (if something weird appears -> treat as invalid)
    if not sub_str.isdigit():
        return np.nan

    frac = int(sub_str) / (10 ** max(1, len(sub_str)))
    return float(sec) + frac


def parse_value_str(v: str) -> float:
    """
    Parse numeric value:
    - accepts both decimal dot and decimal comma
    - rejects text (returns NaN)
    - rejects empty
    """
    s = str(v).replace('"', '').strip()
    if s == "":
        return np.nan

    # allow comma-decimal
    if "," in s and "." not in s:
        s = s.replace(",", ".")

    # now must be numeric
    return pd.to_numeric(s, errors="coerce")


def read_csv_clean_rows(path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Read CSV as raw rows and DROP corrupted rows.

    Rules:
      - ignore col0
      - col1 must be time 'sec,subsec'
      - col2+ must be numeric (accept '.' and ',' decimals)
      - if any value missing or non-numeric -> drop the entire row
      - enforce consistent number of value columns within a file
    """
    kept_rows: list[list] = []
    stats = {
        "lines_total": 0,
        "rows_kept": 0,
        "rows_dropped": 0,
        "drop_bad_parse": 0,
        "drop_wrong_cols": 0,
        "drop_bad_time": 0,
        "drop_bad_value": 0,
    }

    expected_n_values: int | None = None

    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for line in f:
            stats["lines_total"] += 1
            line = line.strip()
            if not line:
                stats["rows_dropped"] += 1
                stats["drop_bad_parse"] += 1
                continue

            # parse per-line to survive broken quoting in single line
            try:
                row = next(csv.reader([line], delimiter=",", quotechar='"'))
            except Exception:
                stats["rows_dropped"] += 1
                stats["drop_bad_parse"] += 1
                continue

            # need at least 3 columns: sensor, time, value1
            if len(row) < 3:
                stats["rows_dropped"] += 1
                stats["drop_wrong_cols"] += 1
                continue

            # take time + values
            time_raw = row[1]
            values_raw = row[2:]

            # enforce consistent number of value columns for this file
            if expected_n_values is None:
                expected_n_values = len(values_raw)
            elif len(values_raw) != expected_n_values:
                stats["rows_dropped"] += 1
                stats["drop_wrong_cols"] += 1
                continue

            t = parse_time_sec_sub_str(time_raw)
            if not np.isfinite(t):
                stats["rows_dropped"] += 1
                stats["drop_bad_time"] += 1
                continue

            vals = [parse_value_str(v) for v in values_raw]
            if any(not np.isfinite(x) for x in vals):
                stats["rows_dropped"] += 1
                stats["drop_bad_value"] += 1
                continue

            kept_rows.append([t] + vals)
            stats["rows_kept"] += 1

    if not kept_rows:
        return pd.DataFrame(), stats

    n_values = len(kept_rows[0]) - 1
    cols = ["t_sec"] + [f"v{i+1}" for i in range(n_values)]
    df = pd.DataFrame(kept_rows, columns=cols)

    # make sure sorted by time (some files might not be strictly sorted)
    df = df.sort_values("t_sec", kind="mergesort").reset_index(drop=True)

    return df, stats


def mode_1(x: pd.Series):
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    m = x.mode()
    return m.iloc[0] if len(m) else np.nan


def aggregate_10s_features_clean(df: pd.DataFrame, window_sec: float = 10.0) -> pd.DataFrame:
    """
    Input df format:
      - t_sec (float seconds)
      - v1..vN numeric
    Windows:
      - anchored to first valid row time
      - [0,10), [10,20), ...
    """
    if df.empty or "t_sec" not in df.columns:
        return pd.DataFrame()

    data_cols = [c for c in df.columns if c != "t_sec"]
    if not data_cols:
        return pd.DataFrame()

    t_sec = df["t_sec"].astype(float)
    t0 = float(t_sec.iloc[0])
    t_rel = t_sec - t0

    window_id = np.floor(t_rel / window_sec).astype("Int64")

    work = df[data_cols].copy()
    work.insert(0, "window_id", window_id)

    g = work.groupby("window_id", dropna=True)

    out = pd.DataFrame({
        "window_start_sec": g["window_id"].first().astype(float) * window_sec,
        "window_end_sec": (g["window_id"].first().astype(float) + 1.0) * window_sec,
        "n_samples": g.size(),
    }).reset_index(drop=True)

    for c in data_cols:
        out[f"{c}_mean"] = g[c].mean().values
        out[f"{c}_median"] = g[c].median().values
        out[f"{c}_std"] = g[c].std(ddof=1).values
        out[f"{c}_min"] = g[c].min().values
        out[f"{c}_max"] = g[c].max().values
        out[f"{c}_mode"] = g[c].apply(mode_1).values

    return out


def process_tree(in_root: Path, out_root: Path):
    processed = 0
    errors = 0
    total_dropped = 0

    for p in in_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() != EXT:
            continue
        if is_ignored(p):
            continue

        rel = p.relative_to(in_root)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            raw_df, stats = read_csv_clean_rows(p)
            feats = aggregate_10s_features_clean(raw_df, window_sec=WINDOW_SEC)
            feats.to_csv(out_path, index=False)
            processed += 1

            total_dropped += stats["rows_dropped"]

            # log a short summary; useful to spot dirty files quickly
            safe_print(
                "[OK]", rel,
                f"rows_kept={stats['rows_kept']}",
                f"rows_dropped={stats['rows_dropped']}",
                f"windows={len(feats)}"
            )

        except Exception as e:
            errors += 1
            safe_print("[ERROR]", rel, "->", e)

    safe_print("\nFinished.")
    safe_print("Processed CSV files:", processed)
    safe_print("Errors:", errors)
    safe_print("Total dropped rows:", total_dropped)
    safe_print("Output folder:", out_root)


def main():
    project_root = Path(__file__).resolve().parent
    in_root = find_in_root(project_root)
    out_root = project_root / OUTPUT_DIRNAME

    safe_print("PROJECT_ROOT:", project_root)
    safe_print("IN_ROOT:", in_root)
    safe_print("OUT_ROOT:", out_root)

    process_tree(in_root, out_root)


if __name__ == "__main__":
    main()
