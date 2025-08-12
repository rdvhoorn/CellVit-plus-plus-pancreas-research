# python scripts/infer_from_filelist.py --filelist_path ./input_data/data_configuration/input_list.csv --output output/full_model_inference/

import argparse
import subprocess
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--filelist_path",
        type=str,
        required=True,
        default=None,
        help="Path to the filelist CSV.",
    )
    parser.add_argument("--classifier", type=Path, default=None)
    parser.add_argument("--resolution", default=0.25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--logdir", type=Path, default=Path("./output/runner_logs"))
    args = parser.parse_args()

    # Build inference command
    command = [
        "../CellViT-plus-plus/cellvit/detect_cells.py",
        "--model",
        str(args.model),
        "--resolution",
        str(args.resolution),
        "--batch_size",
        str(args.batch_size),
        "--geojson",
        "--graph",
        "--outdir",
        str(args.output),
    ]

    # Add classifier or binary flag BEFORE subcommand
    if args.classifier:
        command += ["--classifier_path", str(args.classifier)]
    else:
        command += ["--binary"]

    # Now the subcommand and its (subparser) args
    command += [
        "process_dataset",
        "--filelist",
        str(args.filelist_path),
    ]

    # Log
    print("[INFO] Submitting SLURM job...")
    print(f"[INFO] Command: {' '.join(command)}")

    # Submit to SLURM
    subprocess.run(["sbatch", "scripts/inference_gpu_runner.sh"] + command)


if __name__ == "__main__":
    main()
