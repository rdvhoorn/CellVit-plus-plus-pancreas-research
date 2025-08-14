import pandas as pd
import numpy as np
from pathlib import Path
from typing import Set, List, Tuple
from sklearn.utils import shuffle
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import time
from datetime import datetime


def stratify_greedy_split(
    label_df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42
) -> Tuple[Set[str], Set[str]]:
    label_matrix = label_df.pivot_table(
        index="wsi_id", columns="label_id", values="count", fill_value=0
    )
    label_matrix = label_matrix.loc[shuffle(label_matrix.index, random_state=seed)]
    num_test = int(round(test_frac * len(label_matrix)))
    test_wsis = _greedy_balanced_split(label_matrix, num_test, seed)
    train_wsis = set(label_matrix.index) - set(test_wsis)
    return set(train_wsis), set(test_wsis)


def stratified_kfold_splits(
    wsis: Set[str], label_df: pd.DataFrame, n_splits: int = 5, seed: int = 123
) -> List[Tuple[Set[str], Set[str]]]:
    label_matrix = label_df[label_df["wsi_id"].isin(wsis)].pivot_table(
        index="wsi_id", columns="label_id", values="count", fill_value=0
    )
    folds = []
    wsis = list(label_matrix.index)
    fold_size = len(wsis) // n_splits
    label_matrix = label_matrix.loc[shuffle(label_matrix.index, random_state=seed)]

    used = set()
    for fold_idx in range(n_splits):
        available = list(set(label_matrix.index) - used)
        size = fold_size if fold_idx < n_splits - 1 else len(available)
        val_wsis = _greedy_balanced_split(
            label_matrix.loc[available], size, seed + fold_idx
        )
        train_wsis = set(label_matrix.index) - set(val_wsis)
        folds.append((train_wsis, set(val_wsis)))
        used.update(val_wsis)
    return folds


def _greedy_balanced_split(
    label_matrix: pd.DataFrame, target_size: int, seed: int
) -> Set[str]:
    label_matrix = label_matrix.copy()
    total_distribution = label_matrix.sum()
    selected = set()
    remaining = set(label_matrix.index)
    current_distribution = pd.Series(0, index=label_matrix.columns, dtype=float)

    for _ in range(target_size):
        best_wsi = None
        best_score = float("inf")
        for wsi_id in remaining:
            temp_dist = current_distribution + label_matrix.loc[wsi_id]
            temp_dist_norm = temp_dist / temp_dist.sum()
            target_dist_norm = total_distribution / total_distribution.sum()
            score = np.sum(np.abs(temp_dist_norm - target_dist_norm))
            if score < best_score:
                best_score = score
                best_wsi = wsi_id
        selected.add(best_wsi)
        current_distribution += label_matrix.loc[best_wsi]
        remaining.remove(best_wsi)

    return selected


def copy_wsi_patches_parallel(
    wsi_ids: Set[str],
    src_img: Path,
    src_lbl: Path,
    out_img: Path,
    out_lbl: Path,
    desc="copy",
    max_workers=8,
    timeout=300,
):
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    total = len(wsi_ids)

    def copy_for_wsi(wsi_id: str, index: int):
        start = time.time()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] ⏳ ({index+1}/{total}) Copying {wsi_id}...")

        img_cmd = f"cp {src_img}/{wsi_id}_patch_*.png {out_img}/ 2>/dev/null || true"
        lbl_cmd = f"cp {src_lbl}/{wsi_id}_patch_*.csv {out_lbl}/ 2>/dev/null || true"

        try:
            subprocess.run(["sh", "-c", img_cmd], timeout=timeout)
            subprocess.run(["sh", "-c", lbl_cmd], timeout=timeout)
            print(f"[{ts}] ✅ Copied {wsi_id} in {time.time()-start:.1f}s")
        except subprocess.TimeoutExpired:
            print(f"[{ts}] ❌ Timeout copying {wsi_id}")
        except Exception as e:
            print(f"[{ts}] ❌ Error copying {wsi_id}: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(copy_for_wsi, wsi_id, i)
            for i, wsi_id in enumerate(sorted(wsi_ids))
        ]
        for _ in as_completed(futures):
            pass

    print(f"✅ Copy done: {desc}")


def generate_fold_csvs(
    out_dir: Path, img_dir: Path, folds: List[Tuple[Set[str], Set[str]]]
):
    for i, (train_wsis, val_wsis) in enumerate(folds):
        fold_dir = out_dir / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        def collect(wsi_ids: Set[str]) -> List[str]:
            names = []
            for wsi in wsi_ids:
                matches = sorted(img_dir.glob(f"{wsi}_patch_*.png"))
                names.extend([m.stem for m in matches])
            return sorted(names)

        train_names = collect(train_wsis)
        val_names = collect(val_wsis)

        pd.Series(train_names).to_csv(fold_dir / "train.csv", index=False, header=False)
        pd.Series(val_names).to_csv(fold_dir / "val.csv", index=False, header=False)
        print(f"✅ Fold {i}: {len(train_names)} train, {len(val_names)} val patches")


def print_fold_label_stats(
    folds: List[Tuple[Set[str], Set[str]]], label_df: pd.DataFrame
):
    label_df = label_df.copy()
    label_df["label_id"] = label_df["label_id"].astype(str)

    for i, (train_wsis, val_wsis) in enumerate(folds):
        print(f"\n📊 Fold {i} label distribution:")

        def summarize(wsis: Set[str], name: str):
            df = label_df[label_df["wsi_id"].isin(wsis)]
            total = df["count"].sum()
            summary = df.groupby("label_id")["count"].sum().sort_index()
            print(f"  🔹 {name} set:")
            for label, count in summary.items():
                pct = 100 * count / total
                print(f"     - Label {label}: {count} ({pct:.1f}%)")
            print(f"     Total: {total} cells")

        summarize(train_wsis, "Train")
        summarize(val_wsis, "Val")


if __name__ == "__main__":
    label_csv = Path("output/classifier_data/wsi_label_counts.csv")
    image_dir = Path("output/classifier_data/images")
    label_dir = Path("output/classifier_data/labels")
    output_root = Path("input_data/classifier_dataset")
    folds_dir = output_root / "splits"
    dry_run = False
    n_folds = 4

    label_df = pd.read_csv(label_csv)
    print("📊 Loaded label counts")

    print("🔀 Splitting into train/test WSIs using greedy stratification...")
    train_wsis, test_wsis = stratify_greedy_split(label_df, test_frac=0.2)
    print(f"Train WSIs: {len(train_wsis)} | Test WSIs: {len(test_wsis)}")

    if not dry_run:
        print("📁 Copying training patches...")
        copy_wsi_patches_parallel(
            train_wsis,
            image_dir,
            label_dir,
            output_root / "train/images",
            output_root / "train/labels",
        )
        print("📁 Copying test patches...")
        copy_wsi_patches_parallel(
            test_wsis,
            image_dir,
            label_dir,
            output_root / "test/images",
            output_root / "test/labels",
        )

        print("🔁 Creating stratified folds from training WSIs...")
        folds = stratified_kfold_splits(train_wsis, label_df, n_splits=n_folds)

        print("📊 Printing label stats per fold...")
        print_fold_label_stats(folds, label_df)

        print("📝 Writing fold CSVs...")
        generate_fold_csvs(folds_dir, output_root / "train/images", folds)

        print("✅ Done.")
