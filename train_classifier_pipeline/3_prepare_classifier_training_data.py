"""
prepare_classifier_training_data.py

This script prepares patch-based training data for fine-tuning a CellViT++ classifier head.
It processes a directory of whole-slide images (WSIs), cell detection outputs, and XML annotations
to generate small labeled image patches centered on regions of interest (ROIs).

Each output patch is stored as:
  - A PNG image (RGB tissue crop)
  - A CSV file with labeled cell coordinates (x, y, class_id), relative to the patch

The process is as follows:
1. Load ROI bounding boxes from a metadata CSV file, one row per WSI.
2. For each WSI (identified by filename stem), match the corresponding:
    - TIFF file (WSI)
    - XML file (annotations with polygon regions)
    - .PT file (detected cell locations from CellViT inference)
3. For each ROI in the WSI:
    - Generate sliding window patches (tile_size x tile_size) with optional stride
    - Filter cells from the .pt file that fall into each patch
    - Assign labels to each cell by checking if it lies within a polygon from the XML
    - Save patch PNG and corresponding labeled CSV, only if at least one cell has a valid label

Output patches are saved in:
  output_dir/images/ → image tiles
  output_dir/labels/ → cell coordinate CSVs

Typical usage:
  - Used prior to classifier head training to generate patch-level labeled datasets
  - Output is compatible with CellViT++ training workflows

TODO: Make a separate csv file that has the label counts per wsi, which can be used to make the splits in the next step

sbatch scripts/inference_cpu_runner.sh train_classifier_pipeline/prepare_classifier_training_data.py
"""

from pathlib import Path
from typing import List, Tuple, Dict, Set
import json
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import xml.etree.ElementTree as ET
import torch
import openslide
from PIL import Image
import sys
import csv
import gc
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


from train_classifier_pipeline.label_mapping import LABEL_MAPPING


def load_cell_predictions(pt_path: Path) -> np.ndarray:
    """
    Load cell predictions from .pt file.

    Returns:
        np.ndarray: Nx2 array with (x, y) coordinates
    """
    pt = torch.load(pt_path, map_location="cpu")
    return pt.positions.numpy()


def load_polygons_from_xml(
    xml_path: Path, auto_fix: bool = True
) -> List[Tuple[str, Polygon]]:
    """
    Load all annotation polygons from an XML file.

    Args:
        xml_path (Path): Path to the annotation XML file
        auto_fix (bool): Whether to attempt to auto-fix invalid polygons

    Returns:
        List of tuples: (label, shapely Polygon)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    polygons = []

    for annotation in root.findall(".//Annotation"):
        label = annotation.get("PartOfGroup", "undefined")
        coords = []

        if "roi" in label:
            continue

        for coord in annotation.find("Coordinates"):
            try:
                x = float(coord.get("X"))
                y = float(coord.get("Y"))
                coords.append((x, y))
            except Exception:
                coords = []
                break

        if len(coords) < 3:
            continue

        try:
            poly = Polygon(coords)
            if poly.is_valid and poly.is_simple and not poly.is_empty:
                polygons.append((label, poly))
            elif auto_fix:
                fixed = poly.buffer(0)
                if fixed.is_valid and not fixed.is_empty:
                    polygons.append((label, fixed))
        except Exception:
            continue

    return polygons


def generate_unique_patch_boxes_from_rois(
    roi_metadata: List[Dict[str, int]], tile_size: int, stride: int
) -> List[Tuple[int, int, int, int]]:
    """
    Generate non-overlapping patches from multiple ROI boxes defined in metadata.

    Args:
        roi_metadata: List of dicts with keys 'x', 'y', 'w', 'h'
        tile_size: Width and height of each patch
        stride: Step size between patches

    Returns:
        List of unique patch boxes as (x_min, y_min, x_max, y_max)
    """
    seen: Set[Tuple[int, int]] = set()
    patch_boxes: List[Tuple[int, int, int, int]] = []

    for roi in roi_metadata:
        x_min = int(roi["x"])
        y_min = int(roi["y"])
        x_max = x_min + int(roi["width"])
        y_max = y_min + int(roi["height"])

        for y in range(y_min, y_max - tile_size + 1, stride):
            for x in range(x_min, x_max - tile_size + 1, stride):
                key = (x, y)
                if key not in seen:
                    seen.add(key)
                    patch_boxes.append((x, y, x + tile_size, y + tile_size))

    return patch_boxes


def extract_patch_from_slide(
    slide: openslide.OpenSlide, patch_box: Tuple[int, int, int, int], level: int = 0
) -> np.ndarray:
    """
    Extract an RGB patch from the WSI.

    Args:
        wsi_path: Path to the WSI (.tiff, .svs, etc.)
        patch_box: (x_min, y_min, x_max, y_max) in level 0 coordinates
        level: OpenSlide pyramid level to extract from

    Returns:
        RGB patch as NumPy array of shape (H, W, 3), dtype=uint8
    """
    x_min, y_min, x_max, y_max = patch_box
    width = x_max - x_min
    height = y_max - y_min

    region = slide.read_region((x_min, y_min), level, (width, height))
    rgb = region.convert("RGB")
    return np.array(rgb)


def filter_cells_in_patch(
    cell_coords: np.ndarray, patch_box: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Filters the cells that lie within the given patch box.

    Args:
        cell_coords: Nx2 array of (x, y) cell coordinates (global WSI coords)
        patch_box: Tuple (x_min, y_min, x_max, y_max)

    Returns:
        Mx2 array of filtered (x, y) coordinates inside the patch
    """
    x_min, y_min, x_max, y_max = patch_box

    mask = (
        (cell_coords[:, 0] >= x_min)
        & (cell_coords[:, 0] < x_max)
        & (cell_coords[:, 1] >= y_min)
        & (cell_coords[:, 1] < y_max)
    )

    return cell_coords[mask]


def assign_labels_to_cells(
    cells: np.ndarray, polygons: List[Tuple[str, Polygon]], label_map: Dict[str, int]
) -> List[Tuple[int, int, int]]:
    """
    Assign a label to each cell by checking which polygon it falls inside.

    Args:
        cells: Nx2 array of (x, y) cell coordinates (global)
        polygons: List of (label_name, shapely Polygon)
        label_map: Dictionary mapping label_name -> class_id (int)

    Returns:
        List of tuples: (x, y, class_id), only for cells that matched a polygon
    """
    labeled_cells = []

    for x, y in cells:
        point = Point(x, y)
        for label_name, polygon in polygons:
            if polygon.contains(point):
                class_id = label_map.get(label_name)
                if class_id is not None:
                    labeled_cells.append((int(x), int(y), class_id))
                break  # Stop at first match

    return labeled_cells


def save_patch_and_labels(
    patch_img: np.ndarray,
    cells_with_labels: List[Tuple[int, int, int]],
    patch_box: Tuple[int, int, int, int],
    output_dir: Path,
    patch_id: str,
):
    """
    Save the patch image and corresponding cell labels.

    Args:
        patch_img: NumPy RGB image array (H, W, 3)
        cells_with_labels: List of (x_abs, y_abs, class_id)
        patch_box: (x_min, y_min, x_max, y_max) of patch (to convert abs → rel)
        output_dir: Root directory where images/ and labels/ will be stored
        patch_id: Unique ID for this patch (used for filenames)
    """
    x_min, y_min, _, _ = patch_box

    # Make sure output directories exist
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Save image
    img = Image.fromarray(patch_img)
    img.save(img_dir / f"{patch_id}.png")

    # Save CSV labels
    csv_path = lbl_dir / f"{patch_id}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for x_abs, y_abs, label_id in cells_with_labels:
            # Correct position for crop
            x_rel = x_abs - x_min
            y_rel = y_abs - y_min
            writer.writerow([x_rel, y_rel, label_id])


def process_single_wsi(
    wsi_path: Path,
    xml_path: Path,
    pt_path: Path,
    roi_metadata: List[Dict[str, int]],
    output_dir: Path,
    label_map: Dict[str, int],
    tile_size: int = 256,
    stride: int = 256,
    level: int = 0,
) -> Dict[int, int]:
    """
    Process one WSI by splitting it into patches, assigning labels from XML annotations,
    and saving labeled cell CSVs and patch PNGs for classifier training.

    Returns:
        Dictionary mapping label_id → count of labeled cells in this WSI
    """
    print(f"🚧 Processing WSI: {wsi_path.name}")

    # Load data
    print("📦 Loading detections...")
    cell_coords = load_cell_predictions(pt_path)

    print("🧠 Loading annotations...")
    polygons = load_polygons_from_xml(xml_path)

    print(f"🖼️ Opening WSI once: {wsi_path.name}")
    slide = openslide.OpenSlide(str(wsi_path))

    print("📐 Generating patch grid...")
    patch_boxes = generate_unique_patch_boxes_from_rois(roi_metadata, tile_size, stride)
    print(f"🔲 Total patches: {len(patch_boxes)}")

    label_counts = Counter()
    saved_patch_count = 0

    for i, patch_box in enumerate(patch_boxes):
        # Step 1: Filter cell detections into patch
        patch_cells = filter_cells_in_patch(cell_coords, patch_box)
        if len(patch_cells) == 0:
            continue

        # Step 2: Assign labels to cells
        labeled_cells = assign_labels_to_cells(patch_cells, polygons, label_map)
        if len(labeled_cells) == 0:
            continue

        # Track label counts
        label_counts.update(label for _, _, label in labeled_cells)

        # Step 3: Extract patch image
        try:
            patch_img = extract_patch_from_slide(slide, patch_box, level=level)
        except Exception as e:
            print(f"⚠️  Failed to extract patch {i}: {e}")
            continue

        # Step 4: Save image and label CSV
        patch_id = f"{wsi_path.stem}_patch_{i:04d}"
        save_patch_and_labels(
            patch_img=patch_img,
            cells_with_labels=labeled_cells,
            patch_box=patch_box,
            output_dir=output_dir,
            patch_id=patch_id,
        )

        saved_patch_count += 1

    print(f"✅ Done. Saved {saved_patch_count} labeled patches.")

    slide.close()
    del cell_coords, polygons, patch_boxes
    gc.collect()

    # ✅ Return label counts for aggregation
    return dict(label_counts)


def run_wsi_job(
    args: Tuple[str, Path, Path, Path, List[dict], dict],
) -> Tuple[str, Dict[int, int]]:
    """
    Wrapper function for multiprocessing: processes a single WSI and returns WSI ID + label counts.
    """
    wsi_id, tiff_file, xml_file, pt_file, rois, config = args

    print(f"\n🚀 Processing {wsi_id}")
    label_counts = process_single_wsi(
        wsi_path=tiff_file,
        xml_path=xml_file,
        pt_path=pt_file,
        roi_metadata=rois,
        output_dir=config["output_dir"],
        label_map=config["label_map"],
        tile_size=config["tile_size"],
        stride=config["stride"],
        level=config["level"],
    )

    return wsi_id, label_counts


def process_all_wsis(
    tiff_dir: Path,
    xml_dir: Path,
    pt_dir: Path,
    roi_csv_path: Path,
    output_dir: Path,
    label_map: Dict[str, int],
    tile_size: int = 256,
    stride: int = 256,
    level: int = 0,
    num_workers: int = os.cpu_count(),
):
    """
    Batch process all WSIs using multiprocessing. Also aggregates per-WSI label statistics.

    Args:
        tiff_dir: Directory with .tiff WSI files
        xml_dir: Directory with .xml annotation files
        pt_dir: Directory with .pt detection files
        roi_csv_path: CSV with metadata containing 'path' and 'rois'
        output_dir: Root output directory
        label_map: Dictionary mapping label names to integer IDs
        tile_size: Patch size
        stride: Patch stride
        level: OpenSlide level
        num_workers: Number of parallel processes
    """
    print("📄 Loading ROI metadata...")
    df = pd.read_csv(roi_csv_path)

    # Prepare job list
    jobs = []
    for pt_file in sorted(pt_dir.glob("*.pt")):
        wsi_id = pt_file.stem.replace("_cells", "")
        tiff_file = next(tiff_dir.glob(f"{wsi_id}.*"), None)
        xml_file = next(xml_dir.glob(f"{wsi_id}.xml"), None)

        if not tiff_file or not xml_file:
            print(f"❌ Missing files for {wsi_id}, skipping.")
            continue

        matched_row = df[df["path"].str.endswith(str(tiff_file))]
        if matched_row.empty:
            print(f"❌ No ROI metadata for {wsi_id}, skipping.")
            continue

        rois = json.loads(matched_row.iloc[0]["rois"])
        jobs.append(
            (
                wsi_id,
                tiff_file,
                xml_file,
                pt_file,
                rois,
                {
                    "output_dir": output_dir,
                    "label_map": label_map,
                    "tile_size": tile_size,
                    "stride": stride,
                    "level": level,
                },
            )
        )

    # Process in parallel
    print(f"🧵 Starting processing with {num_workers} workers...")
    all_label_counts = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_wsi_job, job) for job in jobs]
        for future in as_completed(futures):
            try:
                wsi_id, label_counts = future.result()
                for label_id, count in label_counts.items():
                    all_label_counts.append(
                        {"wsi_id": wsi_id, "label_id": label_id, "count": count}
                    )
            except Exception as e:
                print(f"❌ Job failed: {e}")

    # Save aggregated label counts
    if all_label_counts:
        label_df = pd.DataFrame(all_label_counts)
        label_df.to_csv(output_dir / "wsi_label_counts.csv", index=False)
        print(
            f"📊 Saved global label summary to: {output_dir / 'wsi_label_counts.csv'}"
        )
    else:
        print("⚠️ No label counts were collected.")


if __name__ == "__main__":
    cellvit_project_path = "../CellViT-plus-plus"
    sys.path.insert(0, cellvit_project_path)

    tiff_dir = Path("input_data/slides/pancreas/tiffs/")
    xml_dir = Path("input_data/slides/pancreas/xmls/")
    pt_dir = Path("output/full_model_inference/pancreas/")
    roi_csv = Path("input_data/data_configuration/input_list_2025-08-13.csv")
    output_dir = Path("output/classifier_data/")

    process_all_wsis(
        tiff_dir=tiff_dir,
        xml_dir=xml_dir,
        pt_dir=pt_dir,
        roi_csv_path=roi_csv,
        output_dir=output_dir,
        label_map=LABEL_MAPPING,
        tile_size=256,
        stride=256,
        level=0,
        num_workers=None,  # or set to os.cpu_count() or 4, 8, etc.
    )
