# python scripts/infer_single_image.py --wsi ./input_data/slides/R_T14-09850_10.tiff --output output/full_model_inference/
# python scripts/infer_single_image.py --wsi ./input_data/slides/RBIO-GC072-HE-01.tiff --output output/full_model_inference/

import argparse
import subprocess
from pathlib import Path
import csv


def generate_filelist(wsi_path: Path, filelist_path: Path, mpp=0.25, magnification=40):
    with open(filelist_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "slide_mpp", "magnification"])
        writer.writerow([str(wsi_path.resolve()), str(mpp), str(magnification)])


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
    parser.add_argument("--resolution", default="0.25")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--logdir", type=Path, default=Path("./output/runner_logs"))
    args = parser.parse_args()

    filelist_path = args.output / f"filelist_{args.wsi.stem}.csv"
    filelist_path.parent.mkdir(parents=True, exist_ok=True)

    # Write one-image filelist
    generate_filelist(args.wsi, filelist_path)

    # Build inference command
    command = [
        "../CellViT-plus-plus/cellvit/detect_cells.py",
        "--model",
        str(args.model),
        "--resolution",
        args.resolution,
        "--batch_size",
        str(args.batch_size),
        "--geojson",
        "--compression",
        "--graph",
        "--outdir",
        str(args.output),
    ]

    # Add classifier or binary flag BEFORE subcommand
    if args.classifier:
        command += ["--classifier_path", str(args.classifier)]
    else:
        command += ["--binary"]

    # Now add the subcommand and its args
    command += ["process_dataset", "--filelist", str(filelist_path)]

    # Log
    print("[INFO] Submitting SLURM job...")
    print(f"[INFO] Command: {' '.join(command)}")

    # Submit to SLURM
    subprocess.run(["sbatch", "scripts/inference_gpu_runner.sh"] + command)


if __name__ == "__main__":
    main()
