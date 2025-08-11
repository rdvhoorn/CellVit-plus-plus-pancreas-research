# python scripts/infer_single_image.py --wsi ./input_data/slides/R_T14-09850_10.tiff --output output/full_model_inference/
# python scripts/infer_single_image.py --wsi ./input_data/slides/RBIO-GC072-HE-01.tiff --output output/full_model_inference/
# python scripts/infer_single_image.py --wsi ./input_data/slides/RBIO-GC072-HE-01.tiff --output output/full_model_inference/ --rois '[{"x":1950,"y":1890,"width":6123,"height":4632}]'

import argparse
import subprocess
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi", required=True, type=Path, help="Path to the input WSI")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for inference results",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/net/beegfs/users/P098864/projects/CellVit-plus-plus-pancreas-research/checkpoints/CellViT-SAM-H-x40-AMP.pth"
        ),
    )
    parser.add_argument("--classifier", type=Path, default=None)
    parser.add_argument("--resolution", default=0.25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--logdir", type=Path, default=Path("./output/runner_logs"))
    parser.add_argument(
        "--rois",
        type=str,
        help=(
            "Regions of interest (ROIs) as a JSON string or path to a JSON file. "
            "Each ROI should be a dictionary with keys 'x', 'y', 'width', and 'height'."
        ),
    )
    args = parser.parse_args()

    # Parse ROIs
    rois = None
    if args.rois:
        try:
            # Check if it's a JSON string
            rois = json.loads(args.rois)
        except json.JSONDecodeError:
            # If not, assume it's a path to a JSON file
            rois_path = Path(args.rois)
            if rois_path.is_file():
                with open(rois_path, "r") as f:
                    rois = json.load(f)
            else:
                raise ValueError(
                    f"Invalid ROIs argument. Must be a JSON string or a path to a JSON file: {args.rois}"
                )

    filelist_path = args.output / f"filelist_{args.wsi.stem}.csv"
    filelist_path.parent.mkdir(parents=True, exist_ok=True)

    # Build inference command
    command = [
        "../CellViT-plus-plus/cellvit/detect_cells.py",
        "--model",
        str(args.model),
        "--resolution",
        str(args.resolution),
        "--batch_size",
        str(args.batch_size),
        "--gpu",
        "0",
        "--geojson",
        "--compression",
        "--graph",
        "--outdir",
        str(args.output),
    ]

    if rois is not None:
        command += ["--rois", json.dumps(rois)]

    # Add classifier or binary flag BEFORE subcommand
    if args.classifier:
        command += ["--classifier_path", str(args.classifier)]
    else:
        command += ["--binary"]

    # Now the subcommand and its (subparser) args
    command += [
        "process_wsi",
        "--wsi_path",
        str(args.wsi),
        "--wsi_properties",
        json.dumps({"slide_mpp": 0.25, "magnification": 40}),
    ]

    # Log
    print("[INFO] Submitting SLURM job...")
    print(f"[INFO] Command: {' '.join(command)}")

    # Submit to SLURM
    subprocess.run(["sbatch", "scripts/inference_gpu_runner.sh"] + command)


if __name__ == "__main__":
    main()
