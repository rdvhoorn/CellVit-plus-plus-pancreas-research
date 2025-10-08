"""
make_wsi_kfold_splits.py

Purpose
-------
Create a fixed TEST split and K stratified TRAIN/VAL folds at the WSI level,
from a directory containing one CSV per WSI. Each CSV has rows like:
    x,y,label_id
    63552,49447,0
    63680,49526,0
Only the 'label_id' column is used; X/Y are ignored.

Input
-----
--csv_dir: directory with files named <wsi_id>.csv, each containing a header with 'label_id'.
The CSVs can be very large (50k+ rows). The script STREAMS each CSV and only keeps
compact per-WSI label counts in memory (not the raw rows), to remain memory-friendly.

Output
------
--out_dir:
  - test_wsis.txt                      # fixed test set, reused across folds
  - fold_0/train_wsis.txt, val_wsis.txt
  - fold_1/train_wsis.txt, val_wsis.txt
  ...
  - fold_{K-1}/train_wsis.txt, val_wsis.txt

Strategy
----------------------------------------------
1) Aggregate:
   - Open each <wsi_id>.csv and count occurrences per label_id.
   - Build a compact table (wsi_id, label_id, count). No raw coordinates are retained.

2) Fixed TEST selection (greedy stratified):
   - Pivot to a WSI*label matrix of counts.
   - Deterministically shuffle WSIs (seed) for tie-breaking stability.
   - Greedily pick 'num_test = round(test_frac * #WSIs)' WSIs so that the selected set’s
     normalized label distribution stays close to the global normalized distribution,
     minimizing L1 distance at each step.
   - TEST = selected; TRAIN-POOL = remaining.

3) K-folds over TRAIN-POOL (mutually exclusive VAL sets):
   - For fold i in [0..K-1]:
       * AVAILABLE = TRAIN-POOL minus all previously chosen VAL WSIs.
       * VAL size = floor(#TRAIN-POOL / K) for all but the last fold; the last fold gets the remainder.
       * Pick VAL_i greedily (same L1 objective, deterministic).
       * TRAIN_i = TRAIN-POOL - VAL_i.
       * Record (TRAIN_i, VAL_i), and mark VAL_i as used.
   - Result: val_0..val_{K-1} are disjoint; TRAIN_i is the complement within TRAIN-POOL.
   - TEST is fixed and excluded from all folds.

CLI
---
Example:
    python make_wsi_kfold_splits.py \
        --csv_dir /path/to/wsi_csvs \
        --out_dir /path/to/out \
        --test_frac 0.20 \
        --n_folds 5 \
        --seed 42

Summary
-------
The script produces a greedy-stratified train/test/val split: choose a balanced, fixed TEST set,
then form K balanced, mutually exclusive validation folds from the remaining WSIs. It writes plain
text lists of WSI ids and prints label-distribution summaries for TEST and each fold's TRAIN/VAL.
"""

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


# ----------------------------
# I/O: stream and aggregate
# ----------------------------


def count_labels_for_wsi(csv_path: Path) -> Dict[str, int]:
    """
    Stream one WSI CSV and count occurrences per label_id.
    Assumes CSV header contains 'label_id' (case-insensitive).
    """
    counts = Counter()
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return counts
        try:
            label_idx = [h.strip().lower() for h in header].index("label_id")
        except ValueError:
            raise RuntimeError(f"'label_id' column not found in {csv_path}")
        for row in reader:
            if not row or len(row) <= label_idx:
                continue
            label = str(row[label_idx]).strip()
            if label != "":
                counts[label] += 1
    return dict(counts)


def build_label_df(csv_dir: Path, glob_pattern: str = "*.csv") -> pd.DataFrame:
    """
    Walk the folder of per-WSI CSVs, stream-count labels per file, build compact dataframe:
      columns: [wsi_id, label_id, count]
    """
    rows = []
    files = sorted(csv_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {csv_dir}")
    for i, csv_path in enumerate(files, 1):
        wsi_id = csv_path.stem
        counts = count_labels_for_wsi(csv_path)
        if not counts:
            print(
                f"⚠️  Empty or no labels in {csv_path.name}; skipping in stratification.",
                file=sys.stderr,
            )
            continue
        for label_id, c in counts.items():
            rows.append((wsi_id, str(label_id), int(c)))
        if i % 50 == 0:
            print(f"… processed {i}/{len(files)} CSVs")
    if not rows:
        raise RuntimeError("No label counts aggregated; check input files.")
    df = pd.DataFrame(rows, columns=["wsi_id", "label_id", "count"])
    df["wsi_id"] = df["wsi_id"].astype(str)
    df["label_id"] = df["label_id"].astype(str)
    df["count"] = df["count"].astype(int)
    return df


# ----------------------------
# Greedy balanced selection
# ----------------------------


def _pivot_counts(
    label_df: pd.DataFrame, wsis_filter: Iterable[str] = None
) -> pd.DataFrame:
    df = (
        label_df
        if wsis_filter is None
        else label_df[label_df["wsi_id"].isin(wsis_filter)]
    )
    lm = df.pivot_table(
        index="wsi_id", columns="label_id", values="count", fill_value=0
    )
    return lm


def _greedy_balanced_split(
    label_matrix: pd.DataFrame, target_size: int, seed: int
) -> Set[str]:
    """
    Greedily pick WSI ids so that the selected subset's normalized label distribution
    stays close (L1 distance) to the overall normalized distribution.
    """
    if target_size <= 0:
        return set()
    lm = label_matrix.copy().astype(float)
    total_distribution = lm.sum(axis=0)
    total_sum = float(total_distribution.sum())
    if total_sum == 0.0:
        rng = np.random.RandomState(seed)
        wsis = list(lm.index)
        rng.shuffle(wsis)
        return set(wsis[:target_size])
    target_dist_norm = total_distribution / total_sum

    selected: Set[str] = set()
    remaining: Set[str] = set(lm.index)
    current_distribution = pd.Series(0.0, index=lm.columns)

    for _ in range(min(target_size, len(remaining))):
        best_wsi = None
        best_score = float("inf")
        for wsi_id in remaining:
            temp_dist = current_distribution + lm.loc[wsi_id]
            s = float(temp_dist.sum())
            temp_dist_norm = (
                (temp_dist / s) if s > 0 else pd.Series(0.0, index=lm.columns)
            )
            score = np.abs(temp_dist_norm - target_dist_norm).sum()
            if float(score) < best_score:
                best_score = float(score)
                best_wsi = wsi_id
        selected.add(best_wsi)
        current_distribution += lm.loc[best_wsi]
        remaining.remove(best_wsi)
    return selected


def stratify_greedy_split(
    label_df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42
) -> Tuple[Set[str], Set[str]]:
    """
    Train/Test split (WSI-level) via greedy balancing. Returns (train_wsis, test_wsis).
    """
    label_matrix = _pivot_counts(label_df)
    label_matrix = label_matrix.loc[shuffle(label_matrix.index, random_state=seed)]
    num_test = int(round(test_frac * len(label_matrix)))
    test_wsis = _greedy_balanced_split(label_matrix, num_test, seed)
    train_wsis = set(label_matrix.index) - set(test_wsis)
    return set(train_wsis), set(test_wsis)


def stratified_kfold_splits(
    wsis: Set[str], label_df: pd.DataFrame, n_splits: int = 5, seed: int = 123
) -> List[Tuple[Set[str], Set[str]]]:
    """
    Create K folds from the provided WSI set:
      - Disjoint VAL folds selected greedily one by one.
      - For fold i: (train_wsis = all - val_i, val_wsis = val_i)
    """
    label_matrix = _pivot_counts(label_df, wsis)
    wsis_list = list(label_matrix.index)
    if len(wsis_list) < n_splits:
        raise RuntimeError(
            f"Number of WSIs ({len(wsis_list)}) smaller than n_splits ({n_splits})."
        )
    fold_size = len(wsis_list) // n_splits
    label_matrix = label_matrix.loc[shuffle(label_matrix.index, random_state=seed)]

    folds: List[Tuple[Set[str], Set[str]]] = []
    used: Set[str] = set()
    for fold_idx in range(n_splits):
        available = list(set(label_matrix.index) - used)
        size = (
            fold_size if fold_idx < n_splits - 1 else len(available)
        )  # last fold gets remainder
        val_wsis = _greedy_balanced_split(
            label_matrix.loc[available], size, seed + fold_idx
        )
        train_wsis = set(label_matrix.index) - set(val_wsis)
        folds.append((train_wsis, set(val_wsis)))
        used.update(val_wsis)
    return folds


# ----------------------------
# Output & summaries
# ----------------------------


@dataclass
class FoldSets:
    train: Set[str]
    val: Set[str]


def write_test_list(out_dir: Path, test_wsis: Set[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_wsis.txt").write_text("\n".join(sorted(test_wsis)) + "\n")
    print(f"📝 Wrote: {out_dir/'test_wsis.txt'}")


def write_fold_lists(out_dir: Path, folds: List[FoldSets]) -> None:
    for i, f in enumerate(folds):
        fold_dir = out_dir / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "train_wsis.txt").write_text("\n".join(sorted(f.train)) + "\n")
        (fold_dir / "val_wsis.txt").write_text("\n".join(sorted(f.val)) + "\n")
    if folds:
        print(f"📝 Wrote fold lists under: {out_dir}")


def _summarize_set(name: str, wsis: Set[str], label_df: pd.DataFrame) -> None:
    df = label_df[label_df["wsi_id"].isin(wsis)]
    total_cells = int(df["count"].sum())
    num_wsis = len(wsis)
    by_label = df.groupby("label_id")["count"].sum().sort_index()
    print(f"\n📊 {name} — WSIs: {num_wsis}, Cells: {total_cells}")
    if total_cells > 0:
        for lbl, cnt in by_label.items():
            pct = 100.0 * cnt / total_cells
            print(f"  • Label {lbl}: {cnt} ({pct:.1f}%)")
    else:
        print("  • No cells.")


def print_summary(
    test_wsis: Set[str], folds: List[FoldSets], label_df: pd.DataFrame
) -> None:
    print("\n========== Split Summary ==========")
    _summarize_set("TEST", test_wsis, label_df)
    for i, f in enumerate(folds):
        _summarize_set(f"FOLD {i} — TRAIN", f.train, label_df)
        _summarize_set(f"FOLD {i} — VAL", f.val, label_df)
    print("===================================\n")


# ----------------------------
# Main orchestration
# ----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create greedy-stratified TEST split and K stratified folds (train/val) from per-WSI CSVs."
    )
    p.add_argument(
        "--csv_dir",
        type=Path,
        required=True,
        help="Directory with one CSV per WSI (filename <wsi_id>.csv).",
    )
    p.add_argument(
        "--out_dir", type=Path, required=True, help="Output directory for lists."
    )
    p.add_argument(
        "--test_frac",
        type=float,
        default=0.20,
        help="Fraction of WSIs for TEST. Default: 0.20",
    )
    p.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds for TRAIN/VAL. Default: 5",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Seed for deterministic tie-breaking."
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"📂 Reading per-WSI CSVs from: {args.csv_dir}")
    label_df = build_label_df(args.csv_dir)
    n_wsi = label_df["wsi_id"].nunique()
    n_labels = label_df["label_id"].nunique()
    print(f"📊 Aggregated label counts for {n_wsi} WSIs and {n_labels} labels.")

    print("🔀 Creating greedy-stratified TEST split...")
    train_wsis, test_wsis = stratify_greedy_split(
        label_df, test_frac=args.test_frac, seed=args.seed
    )
    print(f"  → Train-pool WSIs: {len(train_wsis)} | Test WSIs: {len(test_wsis)}")

    print(f"🔁 Creating {args.n_folds} stratified folds from training WSIs...")
    folds_raw = stratified_kfold_splits(
        train_wsis, label_df, n_splits=args.n_folds, seed=args.seed + 100
    )
    folds = [FoldSets(train=tr, val=va) for (tr, va) in folds_raw]

    print("📝 Writing lists...")
    write_test_list(args.out_dir, test_wsis)
    write_fold_lists(args.out_dir, folds)

    print("📈 Summary:")
    print_summary(test_wsis, folds, label_df)

    print("✅ Done.")


if __name__ == "__main__":
    main()
