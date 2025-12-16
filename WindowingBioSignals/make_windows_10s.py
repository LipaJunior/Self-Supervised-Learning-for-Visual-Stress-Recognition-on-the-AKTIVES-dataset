from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd


# =======================
# SETTINGS
# =======================
WINDOW_SEC = 10.0
OUTPUT_DIRNAME = "data_windowed_10s_features"
EXT = ".csv"

# ignore facial data completely
IGNORE_IF_PATH_CONTAINS = ["facial"]


# --- make Windows console robust to UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def is_ignored(path: Path) -> bool:
    """Check whether file path should be ignored (e.g. facial data)."""
    low = str(path).lower()
    return any(key in low for key in IGNORE_IF_PATH_CONTAINS)


def safe_print(*args):
    """Print safely even if console encoding is broken."""
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


def read_csv_strict(path: Path) -> pd.DataFrame:
    """
    Read CSV strictly:
    - everything as string
    - no NaN auto-detection
    - safe for 'sec,subsec' timestamps and quoted values
    """
    return pd.read_csv(
        path,
        engine="python",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quotechar='"',
        sep=",",
    )


def parse_time_sec_sub(series: pd.Series) -> pd.Series:
    """
    Parse time formatted as 'seconds,subseconds'.

    Subseconds are interpreted as fractional seconds based on number of digits:
        '84641' -> 0.084641
    """
    s = series.astype(str).str.replace('"', '', regex=False).str.strip()
    parts = s.str.split(",", n=1, expand=True)

    sec = pd.to_numeric(parts[0], errors="coerce")
    sub = parts[1].fillna("0")

    sub_digits = sub.str.len().clip(lower=1)
    sub_num = pd.to_numeric(sub, errors="coerce").fillna(0)

    frac = sub_num / (10 ** sub_digits)
    return sec + frac


def parse_numeric_maybe_comma_decimal(series: pd.Series) -> pd.Series:
    """
    Parse numeric values that may use comma as decimal separator, e.g. "0,123".
    Steps:
      - remove quotes
      - strip spaces
      - replace ',' with '.' (decimal comma -> decimal dot)
      - convert to numeric
    """
    s = series.astype(str).str.replace('"', '', regex=False).str.strip()

    # If there are commas, interpret them as decimal commas for data columns
    # (we do NOT apply this to the time column)
    if s.str.contains(",", regex=False).any():
        s = s.str.replace(",", ".", regex=False)

    return pd.to_numeric(s, errors="coerce")


def mode_1(x: pd.Series):
    """Return first mode or NaN."""
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    m = x.mode()
    return m.iloc[0] if len(m) else np.nan


def aggregate_10s_features(df: pd.DataFrame, window_sec: float = 10.0) -> pd.DataFrame:
    """
    Assumptions:
      - column 1: ignored
      - column 2: time 'sec,subsec'
      - column 3+: data columns (may be quoted and may use decimal comma)

    Windows:
      - anchor to FIRST ROW time
      - relative time = t - t0
      - windows: [0,10), [10,20), ...
    """
    if df.shape[1] < 3:
        return pd.DataFrame()

    time_col = df.columns[1]
    data_cols = list(df.columns[2:])

    t_sec = parse_time_sec_sub(df[time_col])

    # anchor windows to first row
    t0 = float(t_sec.iloc[0])
    t_rel = t_sec - t0

    window_id = np.floor(t_rel / window_sec).astype("Int64")

    work = pd.DataFrame({"window_id": window_id})

    # robust numeric parsing for data columns (handles "0,123")
    for c in data_cols:
        work[c] = parse_numeric_maybe_comma_decimal(df[c])

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
            df = read_csv_strict(p)
            feats = aggregate_10s_features(df, window_sec=WINDOW_SEC)
            feats.to_csv(out_path, index=False)
            processed += 1

            if len(feats) > 0:
                safe_print(
                    "[OK]", rel,
                    "windows:", len(feats),
                    "first window:", feats.iloc[0]["window_start_sec"],
                    "-", feats.iloc[0]["window_end_sec"]
                )
            else:
                safe_print("[OK]", rel, "(no windows)")

        except Exception as e:
            errors += 1
            safe_print("[ERROR]", rel, "->", e)

    safe_print("\nFinished.")
    safe_print("Processed CSV files:", processed)
    safe_print("Errors:", errors)
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
