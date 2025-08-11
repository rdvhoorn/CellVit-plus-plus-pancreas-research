import snappy

in_path = "output/full_model_inference/RBIO-GC072-HE-01_cells.geojson.snappy"
out_path = "output/full_model_inference/RBIO-GC072-HE-01_cells.geojson"

with open(in_path, "rb") as f_in:
    compressed = f_in.read()

decompressed = snappy.uncompress(compressed)

with open(out_path, "wb") as f_out:
    f_out.write(decompressed)
