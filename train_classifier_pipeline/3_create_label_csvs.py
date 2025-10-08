from pathlib import Path
import sys
from shapely.geometry import Point, Polygon
from concurrent.futures import ProcessPoolExecutor, as_completed
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
import os
from shapely.prepared import prep
from typing import Dict, List, Tuple

# Import your label map
from train_classifier_pipeline.label_mapping import LABEL_MAPPING


def load_cell_predictions(pt_path: Path):
    """Load cell detections from .pt file → Nx2 NumPy array."""
    pt = torch.load(pt_path, map_location="cpu")
    return pt.positions.numpy().astype(int)


def load_polygons_from_xml(xml_path: Path) -> List[Tuple[str, Polygon]]:
    """Load all annotation polygons from XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polygons = []

    for annotation in root.findall(".//Annotation"):
        label = annotation.get("PartOfGroup", "undefined")
        if "roi" in label.lower():
            continue

        coords = []
        for coord in annotation.find("Coordinates"):
            try:
                x = float(coord.get("X"))
                y = float(coord.get("Y"))
                coords.append((x, y))
            except Exception:
                coords = []
                break

        if len(coords) >= 3:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_valid and not poly.is_empty:
                polygons.append((label, poly))

    return polygons


def assign_labels(cells, polygons, label_map: Dict[str, int]):
    """Assign label_id to each cell if it falls in a polygon."""
    results = []
    for x, y in cells:
        p = Point(x, y)
        for label_name, poly in polygons:
            if poly.contains(p):
                label_id = label_map.get(label_name)
                if label_id is not None:
                    results.append((x, y, label_id))
                break
    return results


def process_single_wsi(
    pt_path: Path, xml_path: Path, label_map: Dict[str, int], output_dir: Path
):
    """Process a single WSI: assign labels and save to CSV."""
    wsi_id = pt_path.stem.replace("_cells", "")
    print(f"🔹 Processing {wsi_id}")

    try:
        csv_path = output_dir / f"{wsi_id}.csv"

        if csv_path.exists():
            print(f"⚠️ {csv_path} already exists, skipping.")
            return

        cells = load_cell_predictions(pt_path)
        polygons = load_polygons_from_xml(xml_path)
        polygons = [
            (label, prep(poly)) for label, poly in polygons if label in label_map
        ]
        labeled = assign_labels(cells, polygons, label_map)

        if not labeled:
            print(f"⚠️ No labeled cells found for {wsi_id}")
            return

        np.savetxt(
            csv_path,
            np.asarray(labeled, dtype=int),
            fmt="%d",
            delimiter=",",
            header="x,y,label_id",
            comments="",  # ✅ disables the '#' prefix
        )
        print(f"✅ Saved {csv_path.name} ({len(labeled)} rows)")

    except Exception as e:
        # Print traceback for debugging
        import traceback

        traceback.print_exc()
        print(f"❌ Error processing {wsi_id}: {e}")


def main(
    xml_dir: Path,
    pt_dir: Path,
    output_dir: Path,
    label_map: Dict[str, int],
    num_workers: int = os.cpu_count() - 2,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(pt_dir.glob("*.pt"))
    jobs = []
    for pt_file in pt_files:
        wsi_id = pt_file.stem.replace("_cells", "")
        xml_file = xml_dir / f"{wsi_id}.xml"
        if not xml_file.exists():
            print(f"❌ Missing XML for {wsi_id}, skipping.")
            continue
        jobs.append((pt_file, xml_file))

    print(f"🧵 Starting {len(jobs)} jobs with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_single_wsi, pt, xml, label_map, output_dir)
            for pt, xml in jobs
        ]
        for _ in as_completed(futures):
            pass

    print("🏁 Done processing all WSIs.")


if __name__ == "__main__":
    cellvit_project_path = "../CellViT-plus-plus"
    sys.path.insert(0, cellvit_project_path)

    xml_dir = Path("/net/beegfs/groups/mmai/cellvit_pancreas/amc_cases/")
    pt_dir = Path("output/full_model_inference/amc_cases/")
    output_dir = Path("output/cell_labels/")
    main(xml_dir, pt_dir, output_dir, LABEL_MAPPING, num_workers=10)
