import xml.etree.ElementTree as ET
import os
import json
import pandas as pd
from datetime import datetime


def extract_roi_rectangles(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rois = []

    for ann in root.findall(".//Annotation"):
        if (
            ann.attrib.get("Type") == "Rectangle"
            and ann.attrib.get("PartOfGroup") == "roi"
        ):
            coords = ann.find("Coordinates")
            xs = [float(c.attrib["X"]) for c in coords.findall("Coordinate")]
            ys = [float(c.attrib["Y"]) for c in coords.findall("Coordinate")]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            width = x_max - x_min
            height = y_max - y_min

            rois.append({"x": x_min, "y": y_min, "width": width, "height": height})

    return rois


def collect_tiff_roi_data(tiff_directory, xml_directory):
    rows = []

    for root, _, files in os.walk(tiff_directory):
        for file in files:
            if file.lower().endswith(".tiff"):
                tiff_path = os.path.join(root, file)
                xml_filename = os.path.splitext(file)[0] + ".xml"
                xml_path = os.path.join(xml_directory, xml_filename)

                if os.path.exists(xml_path):
                    rois = extract_roi_rectangles(xml_path)
                    rows.append(
                        {
                            "path": tiff_path,
                            "slide_mpp": 0.25,
                            "magnification": 40,
                            "rois": json.dumps(rois),
                        }
                    )

    return pd.DataFrame(rows)


df = collect_tiff_roi_data(
    tiff_directory="/net/beegfs/groups/mmai/cellvit_pancreas/amc_cases/",
    xml_directory="/net/beegfs/groups/mmai/cellvit_pancreas/amc_cases/",
)
print(df)

date_str = datetime.now().strftime("%Y-%m-%d")
output_path = f"input_data/data_configuration/input_list_{date_str}.csv"

df.to_csv(output_path, index=False)
