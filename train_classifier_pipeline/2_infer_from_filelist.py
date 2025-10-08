# python scripts/infer_from_filelist.py --filelist_path ./input_data/data_configuration/input_list.csv --output output/full_model_inference/
# python scripts/infer_from_filelist.py --filelist_path ./input_data/data_configuration/input_list_2025-08-13.csv --output output/full_model_inference/pancreas

import argparse
import subprocess
from pathlib import Path
import pandas as pd
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
        help="Path to the filelist CSV.",
    )
    parser.add_argument("--classifier", type=Path, default=None)
    parser.add_argument("--resolution", default=0.25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--logdir", type=Path, default=Path("./output/runner_logs"))
    parser.add_argument(
        "--split_jobs",
        action="store_true",
        help="If set, submit one sbatch job per WSI in the filelist instead of one dataset job",
    )
    args = parser.parse_args()

    if not args.split_jobs:
        # Original behavior: one job for whole dataset
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
            "--compression",
            "--outdir",
            str(args.output),
        ]

        if args.classifier:
            command += ["--classifier_path", str(args.classifier)]
        else:
            command += ["--binary"]

        command += [
            "process_dataset",
            "--filelist",
            str(args.filelist_path),
        ]

        print("[INFO] Submitting SLURM job...")
        print(f"[INFO] Command: {' '.join(command)}")
        subprocess.run(["sbatch", "scripts/inference_gpu_runner.sh"] + command)

    else:
        # New behavior: one job per WSI
        df = pd.read_csv(args.filelist_path)
        for _, row in df.iterrows():
            wsi_path = Path(row["path"])
            slide_mpp = row.get("slide_mpp", args.resolution)
            magnification = row.get("magnification", 40)
            rois = row.get("rois", None)

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

            if args.classifier:
                command += ["--classifier_path", str(args.classifier)]
            else:
                command += ["--binary"]

            command += [
                "process_wsi",
                "--wsi_path",
                str(wsi_path),
                "--wsi_properties",
                json.dumps({"slide_mpp": slide_mpp, "magnification": magnification}),
            ]

            if isinstance(rois, str) and rois.strip():
                command += ["--rois", rois]

            print("[INFO] Submitting SLURM job...")
            print(f"[INFO] Command: {' '.join(command)}")
            subprocess.run(["sbatch", "scripts/inference_gpu_runner.sh"] + command)


if __name__ == "__main__":
    main()
