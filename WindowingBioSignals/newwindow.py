from __future__ import annotations

from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd

# =======================
# SETTINGS
# =======================
WINDOW_LEN_SEC = 30.0  # offline context: [T-30, T]
OUTPUT_DIRNAME = f"data_label_aligned_{int(WINDOW_LEN_SEC)}s"
EXT = ".csv"

IGNORE_IF_PATH_CONTAINS = ["facial"]

# Match sensor files by name (case-insensitive, flexible)
ACC_RE = re.compile(r"\bacc\b", re.IGNORECASE)
BVP_RE = re.compile(r"\bbvp\b|\bppg\b", re.IGNORECASE)
EDA_RE = re.compile(r"\beda\b|\bgsr\b", re.IGNORECASE)

# --- make Windows console robust to UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def safe_print(*args):
    try:
        print(*args)
    except UnicodeEncodeError:
        msg = " ".join(str(a) for a in args).encode("utf-8", "replace").decode("utf-8", "replace")
        print(msg)


def is_ignored(path: Path) -> bool:
    low = str(path).lower()
    return any(key in low for key in IGNORE_IF_PATH_CONTAINS)


def find_data_root(project_root: Path) -> Path:
    candidates = []
    for p in project_root.rglob("DATA"):
        if p.is_dir():
            csv_count = sum(1 for f in p.rglob("*.csv") if not is_ignored(f))
            candidates.append((csv_count, p))
    if not candidates:
        raise FileNotFoundError(f"DATA folder not found inside project: {project_root}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def read_csv_strict(path: Path) -> pd.DataFrame:
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
    s = series.astype(str).str.replace('"', '', regex=False).str.strip()
    parts = s.str.split(",", n=1, expand=True)

    sec = pd.to_numeric(parts[0], errors="coerce")
    sub = parts[1].fillna("0")

    sub_digits = sub.str.len().clip(lower=1)
    sub_num = pd.to_numeric(sub, errors="coerce").fillna(0)

    frac = sub_num / (10 ** sub_digits)
    return sec + frac


def parse_numeric_maybe_comma_decimal(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace('"', '', regex=False).str.strip()
    if s.str.contains(",", regex=False).any():
        s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def majority_vote(values: list[str]) -> str | float:
    values = [v for v in values if isinstance(v, str) and v.strip() != ""]
    if not values:
        return np.nan
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def load_labels(labels_csv: Path) -> pd.DataFrame:
    # Try 2-row header first
    try:
        df = pd.read_csv(labels_csv, header=[0, 1], engine="python", dtype=str)
        df.columns = [
            (str(a).strip() if a is not None else "") + "__" + (str(b).strip() if b is not None else "")
            for a, b in df.columns
        ]
    except Exception:
        df = pd.read_csv(labels_csv, header=0, engine="python", dtype=str)
        df.columns = [str(c).strip() for c in df.columns]

    def pick_contains(all_substrings: list[str]) -> str | None:
        for c in df.columns:
            low = c.lower()
            if all(s.lower() in low for s in all_substrings):
                return c
        return None

    minute_col = pick_contains(["minute"])
    second_col = pick_contains(["second"])
    if second_col is None:
        raise ValueError("Labels: could not find a 'Second' column.")

    minute = parse_numeric_maybe_comma_decimal(df[minute_col]) if minute_col else pd.Series(0, index=df.index)
    second = parse_numeric_maybe_comma_decimal(df[second_col])

    # If seconds look cumulative (e.g., 10,20,30,... or 70,80,...), use directly
    if np.nanmax(second.values) > 59:
        time_sec = second
    else:
        time_sec = minute * 60.0 + second

    e1_stress = pick_contains(["expert1", "stress"])
    e2_stress = pick_contains(["expert2", "stress"])
    e3_stress = pick_contains(["expert3", "stress"])
    e1_react = pick_contains(["expert1", "reaction"])
    e2_react = pick_contains(["expert2", "reaction"])
    e3_react = pick_contains(["expert3", "reaction"])

    # fallback if column names are not multiheaders
    if e1_stress is None:
        e1_stress = pick_contains(["stress/no stress"])
    if e1_react is None:
        e1_react = pick_contains(["reaction/no reaction"])

    stress_label = df.apply(
        lambda r: majority_vote([
            r[e1_stress] if e1_stress else "",
            r[e2_stress] if e2_stress else "",
            r[e3_stress] if e3_stress else "",
        ]),
        axis=1
    )

    reaction_label = df.apply(
        lambda r: majority_vote([
            r[e1_react] if e1_react else "",
            r[e2_react] if e2_react else "",
            r[e3_react] if e3_react else "",
        ]),
        axis=1
    )

    out = pd.DataFrame({
        "time_sec": time_sec.astype(float),
        "stress_label": stress_label,
        "reaction_label": reaction_label,
    })

    out = out.dropna(subset=["time_sec"])
    out = out[(out["stress_label"].notna()) | (out["reaction_label"].notna())].reset_index(drop=True)
    return out


def extract_label_aligned_features(df: pd.DataFrame, label_times_sec: pd.Series, window_len_sec: float) -> pd.DataFrame:
    if df.shape[1] < 3:
        return pd.DataFrame()

    time_col = df.columns[1]
    data_cols = list(df.columns[2:])

    t_abs = parse_time_sec_sub(df[time_col])
    t0 = float(t_abs.iloc[0])
    t = t_abs - t0  # relative to start

    parsed = {c: parse_numeric_maybe_comma_decimal(df[c]) for c in data_cols}
    parsed = pd.DataFrame(parsed)

    rows = []
    for T in label_times_sec.astype(float).values:
        start = float(T - window_len_sec)
        end = float(T)
        mask = (t >= start) & (t <= end)
        n = int(mask.sum())
        if n < 3:
            continue

        row = {
            "label_time_sec": float(T),
            "window_start_sec": start,
            "window_end_sec": end,
            "n_samples": n,
        }

        for c in data_cols:
            x = parsed.loc[mask, c].dropna()
            row[f"{c}_mean"] = float(x.mean()) if len(x) else np.nan
            row[f"{c}_std"] = float(x.std(ddof=1)) if len(x) > 1 else (0.0 if len(x) == 1 else np.nan)
            row[f"{c}_min"] = float(x.min()) if len(x) else np.nan
            row[f"{c}_max"] = float(x.max()) if len(x) else np.nan
            row[f"{c}_median"] = float(x.median()) if len(x) else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def detect_signal_type(path: Path) -> str | None:
    name = path.stem  # without extension
    if ACC_RE.search(name):
        return "ACC"
    if BVP_RE.search(name):
        return "BVP"
    if EDA_RE.search(name):
        return "EDA"
    return None


def debug_print_data_tree(data_root: Path, max_lines: int = 80):
    safe_print("\n--- DEBUG: top of DATA tree ---")
    lines = 0
    for p in sorted(data_root.rglob("*")):
        if lines >= max_lines:
            safe_print("... (truncated)")
            break
        rel = p.relative_to(data_root)
        if p.is_dir():
            safe_print("[DIR ]", rel)
            lines += 1
        elif p.suffix.lower() == ".csv":
            safe_print("[CSV ]", rel)
            lines += 1
    safe_print("--- END DEBUG ---\n")


def find_sessions_with_labels(data_root: Path) -> list[Path]:
    """
    A 'session' = any directory that contains ExpertLabels.csv
    (This matches your child/game folders even if structure varies.)
    """
    sessions = []
    for lbl in data_root.rglob("ExpertLabels.csv"):
        if lbl.is_file() and not is_ignored(lbl):
            sessions.append(lbl.parent)
    return sorted(set(sessions))


def process_dataset(data_root: Path, out_root: Path):
    processed = 0
    skipped_sessions_no_signals = 0
    errors = 0

    debug_print_data_tree(data_root)

    sessions = find_sessions_with_labels(data_root)
    safe_print("Found label sessions:", len(sessions))
    if not sessions:
        safe_print("No ExpertLabels.csv found under DATA. Check filename/case.")
        return

    for sess_dir in sessions:
        rel_sess = sess_dir.relative_to(data_root)
        labels_path = sess_dir / "ExpertLabels.csv"

        try:
            labels = load_labels(labels_path)
        except Exception as e:
            errors += 1
            safe_print("[ERROR LABELS]", rel_sess, "->", e)
            continue

        # Find any CSV signals in this session folder
        sensor_csvs = []
        for f in sess_dir.glob("*.csv"):
            if f.name == "ExpertLabels.csv":
                continue
            if is_ignored(f):
                continue
            sig = detect_signal_type(f)
            if sig is not None:
                sensor_csvs.append((sig, f))

        if not sensor_csvs:
            skipped_sessions_no_signals += 1
            safe_print("[SKIP SESSION - NO SIGNAL FILES]", rel_sess)
            continue

        out_sess_dir = out_root / rel_sess
        out_sess_dir.mkdir(parents=True, exist_ok=True)

        for sig, fpath in sensor_csvs:
            try:
                df = read_csv_strict(fpath)
                feats = extract_label_aligned_features(df, labels["time_sec"], WINDOW_LEN_SEC)
                if len(feats) == 0:
                    safe_print("[OK]", rel_sess, fpath.name, "(no aligned windows)")
                    feats.to_csv(out_sess_dir / f"{sig}_features.csv", index=False)
                    processed += 1
                    continue

                merged = feats.merge(labels, left_on="label_time_sec", right_on="time_sec", how="left")
                merged = merged[(merged["stress_label"].notna()) | (merged["reaction_label"].notna())].reset_index(drop=True)

                merged.to_csv(out_sess_dir / f"{sig}_features.csv", index=False)
                processed += 1
                safe_print("[OK]", rel_sess, fpath.name, "->", f"{sig}_features.csv", "rows:", len(merged))
            except Exception as e:
                errors += 1
                safe_print("[ERROR]", rel_sess, fpath.name, "->", e)

    safe_print("\nFinished.")
    safe_print("Processed signal files:", processed)
    safe_print("Skipped sessions (no signals):", skipped_sessions_no_signals)
    safe_print("Errors:", errors)
    safe_print("Output folder:", out_root)


def main():
    project_root = Path(__file__).resolve().parent
    data_root = find_data_root(project_root)
    out_root = project_root / OUTPUT_DIRNAME
    out_root.mkdir(parents=True, exist_ok=True)

    safe_print("PROJECT_ROOT:", project_root)
    safe_print("DATA_ROOT:", data_root)
    safe_print("OUT_ROOT:", out_root)
    safe_print("WINDOW_LEN_SEC:", WINDOW_LEN_SEC)

    process_dataset(data_root, out_root)


if __name__ == "__main__":
    main()
