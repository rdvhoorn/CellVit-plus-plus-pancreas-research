import os
from pathlib import Path
import sys

from wsi_patching.core import WSIGrid, PatchExtractor
from wsi_patching.regions_of_interest import AttachROIs, RectROIfromXMLProvider
from wsi_patching.annotations import AddCellAnnotationFromCSV
from wsi_patching.encoders import PNGEncoder
from wsi_patching.writers import WebDatasetWriter


def main(wsi_dir, roi_dir, label_dir, output_dir):
    # A list of all the .tiff files in the wsi_dir
    csvs = sorted(label_dir.glob("*.csv"))

    # Get the stems of the files 'as file ids'
    stems = [str(s.stem) for s in csvs]

    # Set slides to strings
    slides = [str(wsi_dir / f"{stem}.tiff") for stem in stems]
    print(f"Using slides: {slides}")

    # Roi dict: {stem: path_to_xml} for each slide
    rois = {stem: roi_dir / f"{stem}.xml" for stem in stems}

    # Label dict: {stem: path_to_csv} for each slide
    labels = {stem: label_dir / f"{stem}.csv" for stem in stems}

    pipeline = (
        WSIGrid(slides=slides, use_gpu=True, level=0)
        .then(AttachROIs(providers=[RectROIfromXMLProvider(rois=rois)]))
        .then(PatchExtractor(tile_size=512, stride=512, 
                             max_batch_size=400, max_window_size=512 * 10))
        .then(AddCellAnnotationFromCSV(wsi_to_csv_mapping=labels, 
                                       label_col="label_id", 
                                       filter_empty=True))
        .then(PNGEncoder(compress_level=6, 
                         threads=20))
        .to(WebDatasetWriter(outdir=output_dir, 
                             shard_size=5000, 
                             shuffle_buffer_size=15000))
    )

    pipeline.run(
        cpu_processes=8,
        verbosity_level="INFO",
        gracefully_handle_producer_errors=False,
    )


if __name__ == "__main__":
    xml_dir = Path("/net/beegfs/groups/mmai/cellvit_pancreas/amc_cases/")
    tiff_dir = Path("/net/beegfs/groups/mmai/cellvit_pancreas/amc_cases/")
    csv_dir = Path("input_data/cell_labels/")
    output_dir = Path("input_data/webdataset/")
    main(tiff_dir, xml_dir, csv_dir, output_dir)
