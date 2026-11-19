---
name: opera-disp-s1-data
description: Get OPERA DISP-S1 (Sentinel-1 InSAR displacement) data for a bounding box and date range as Zarr stacks with `opera-utils disp-s1-stack`, and read it correctly. Use when the user asks for OPERA DISP or DISP-S1 data, Sentinel-1 displacement time series over an area, or a comparison against OPERA products.
---

# OPERA DISP-S1 data for a region

Use one command. Do not chain `disp-s1-intersects`, `disp-s1-download` and `disp-s1-reformat` by hand unless `disp-s1-stack` is missing.

## Steps

1. List the frames that cover the region. This queries CMR and downloads nothing.

   ```bash
   opera-utils disp-s1-stack --bbox W S E N --start-datetime YYYY-MM-DD --list-only
   ```

   The table gives each frame ID, orbit pass, product count, and first and last date. A small region is often inside two ascending and two descending frames from overlapping tracks. The frames end on different dates.

2. Download, crop and reformat.

   ```bash
   opera-utils disp-s1-stack --bbox W S E N \
       --start-datetime YYYY-MM-DD [--end-datetime YYYY-MM-DD] \
       --output-dir DIR [--frame-ids ID ...] [--orbit-pass ASCENDING|DESCENDING]
   ```

   Outputs in `DIR`:
   - `F<frame>/`: the cropped NetCDF products.
   - `F<frame>.zarr`: one stack per frame.
   - `stack_manifest.json`: the request, the frames (ID, orbit pass, dates, files) and the opera-utils version.

3. Open a stack with `xr.open_zarr(path, consolidated=False)`.
   - `displacement` is line-of-sight (LOS) motion in meters, rebased to the first date of the frame. Negative is away from the satellite (for example, subsidence).
   - The quality layers (`temporal_coherence`, `phase_similarity`, `recommended_mask`, `connected_component_labels`, `water_mask` and others) stay in the stack.
   - The CRS is in `spatial_ref.attrs["crs_wkt"]`.

4. To extend the stacks, run the same command with a later `--end-datetime`. Complete subsets are kept. A stack is rebuilt only when its file list changes. `--overwrite` rebuilds all stacks.

The Python API is `from opera_utils.disp import make_stack`. It takes the same arguments.

## Time and size

- Over HTTPS, each product (about 400 MB) downloads in full and is then cropped. A frame from 2016 to now has about 400 products (about 160 GB). At 100 MB/s that is about 30 minutes per frame. Use a narrow date range or `--frame-ids` when you can.
- In the AWS us-west-2 region, `--url-type S3` reads only the crop.
- Credentials: Earthdata Login for `urs.earthdata.nasa.gov` in `~/.netrc`, or `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`.

## Rules

- **Do not mask by default.** The products have values at almost every land pixel. `recommended_mask` marks low-quality values, and the default `disp-s1-stack` run does not apply it. If data looks missing (for example in the "filtered" view of the ASF Vertex viewer), check the unmasked stack before you say the product has no data. Report quality from the quality layers.
- **Use the manifest for the frame direction.** `orbit_pass` comes from the frame database. A frame label copied from a web viewer can be wrong.
- **Do not average overlapping frames.** Frames from different tracks have different incidence angles, dates and references.
- **Re-reference before you compare.** Each stack is relative to its own reference (default: median of the high-coherence pixels). Subtract a local stable area before you compare frames, tracks or sensors.
- **A flat series is not proof of stability.** DISP-S1 is C-band (5.5 cm wavelength) at 30 m. Unwrapping needs the LOS change between neighboring pixels to stay below a quarter wavelength (about 1.4 cm) per revisit. The total motion in one interferogram is not the limit. Fast or steep deformation, or decorrelation, can make the series flat or wrong. Check `temporal_coherence` and `recommended_mask` before you trust a value.

## If `disp-s1-stack` is missing

The installed opera-utils is too old. Run these per frame:

```bash
opera-utils disp-s1-intersects --bbox W S E N --ids-only
opera-utils disp-s1-download --frame-id ID --bbox W S E N --output-dir DIR/F<ID>
opera-utils disp-s1-reformat --input-files DIR/F<ID>/*.nc --output-name DIR/F<ID>.zarr \
    --quality-datasets None --quality-thresholds None
```

Older `disp-s1-download` versions can leave a truncated file after an interrupted run, and a later run skips it as existing. Open each file before you reformat, and delete any that fail.
