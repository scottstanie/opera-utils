import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyproj

from opera_utils import _disp
from opera_utils.credentials import get_earthaccess_s3_creds


def main():
    """Use parallel DispReader to create CSV output."""
    parser = argparse.ArgumentParser(description="Read OPERA DISP-S1 files in parallel")
    parser.add_argument(
        "url_file",
        type=str,
        help="File containing S3 URLs to OPERA DISP-S1 files, one per line",
    )
    parser.add_argument(
        "--row-col",
        type=int,
        nargs=2,
        required=True,
        help="(row, column) within the frame to read from.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of worker processes to use",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=4 * 1024 * 1024,
        help="Page size in bytes for HDF5 file system page strategy",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--sample-locations",
        type=int,
        default=3,
        help="Number of sample locations to read",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate time series plots",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read URLs from file
    with open(args.url_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Found {len(urls)} URLs to process")

    # Get AWS credentials for accessing the files
    aws_credentials = get_earthaccess_s3_creds("opera-uat")

    # Create and use the reader
    t0 = time.time()
    reader = _disp.DispReader(
        urls,
        page_size=args.page_size,
        max_workers=args.max_workers,
        use_multiprocessing=args.max_workers > 1,
        aws_credentials=aws_credentials,
    )

    # Get the metadata object for the current frame
    disp_file = reader.disp_files[0]
    frame_metadata = _disp.FrameMetadata.from_frame_id(disp_file.frame_id)
    if len(set(df.frame_id for df in reader.disp_files)) > 1:
        raise ValueError("More than one frame found in reader")

    # Set up the transformer from row/col to UTM and lat/lon
    from rasterio.transform import Affine, AffineTransformer

    transform = Affine.from_gdal(*frame_metadata.geotransform)
    rowcol_to_utm = AffineTransformer(transform)
    utm_to_lonlat = pyproj.Transformer.from_crs(
        frame_metadata.crs.to_epsg(), "EPSG:4326", always_xy=True
    )

    t0 = time.time()
    reader.open()
    print(f".open() run in {time.time() - t0:.2f} seconds")

    # Get the row and column from arguments
    base_row, base_col = args.row_col

    # Define sample locations - different offsets from the base row/col
    sample_offsets = [
        (0, 0),  # Base location
        (10, 10),  # Offset diagonally by 10 pixels
        (100, 0),  # Offset vertically by 100 pixels
    ]

    # Add more sample locations if requested
    if args.sample_locations > len(sample_offsets):
        for i in range(len(sample_offsets), args.sample_locations):
            # Generate some semi-random offsets
            row_offset = (i * 37) % 200  # Use prime numbers to avoid patterns
            col_offset = (i * 23) % 200
            sample_offsets.append((row_offset, col_offset))

    # Get the unique dates for our time series
    # Skip the first date as it's the reference (displacement = 0)
    dates = reader.unique_dates[1:]
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    # Prepare to collect all time series data
    all_series_data = []

    # Read time series at each location
    for i, (row_offset, col_offset) in enumerate(sample_offsets):
        row = base_row + row_offset
        col = base_col + col_offset

        t0 = time.time()
        # Read just 1 pixel at the specified location
        # The shape will be (time_steps, 1, 1)
        data = reader[:, row : row + 1, col : col + 1]
        read_time = time.time() - t0
        print(
            f"Location {i + 1}: ({row}, {col}) - Read shape {data.shape} in {read_time:.2f} seconds"
        )

        # Extract the time series (squeeze removes singleton dimensions)
        time_series = data.squeeze()

        utm_x, utm_y = rowcol_to_utm.xy([row], [col])
        lon, lat = utm_to_lonlat.transform(utm_x, utm_y, radians=False)
        # Create a dictionary for this location's data
        series_data = {
            "location_id": i + 1,
            "row": row,
            "col": col,
            "lon": lon,
            "lat": lat,
            "x": utm_x,
            "y": utm_y,
            "time_series": time_series,
        }
        all_series_data.append(series_data)

    # Close the reader
    reader.close()

    # Create a DataFrame with all time series
    # First, prepare the data in wide format
    df_data = {"date": date_strs}

    for series in all_series_data:
        location_id = series["location_id"]
        row = series["row"]
        col = series["col"]
        time_series = series["time_series"]

        # Add this location's time series as a column
        column_name = f"disp_loc{location_id}_r{row}_c{col}"
        df_data[column_name] = time_series

    # Create the DataFrame
    df = pd.DataFrame(df_data)

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = output_dir / f"displacement_timeseries_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Time series data saved to {csv_filename}")

    # Also save in "long" format which might be better for some analyses
    df_long = df.melt(id_vars=["date"], var_name="location", value_name="displacement")

    # Extract location metadata from column names
    df_long[["loc_id", "row", "col"]] = df_long["location"].str.extract(
        r"disp_loc(\d+)_r(\d+)_c(\d+)"
    )

    # Convert to numeric
    for col in ["loc_id", "row", "col"]:
        df_long[col] = pd.to_numeric(df_long[col])

    # Save long format to CSV
    long_csv_filename = output_dir / f"displacement_timeseries_long_{timestamp}.csv"
    df_long.to_csv(long_csv_filename, index=False)
    print(f"Long-format time series data saved to {long_csv_filename}")

    # Create metadata file with information about the dataset
    metadata = {
        "base_location": f"({base_row}, {base_col})",
        "number_of_locations": len(sample_offsets),
        "number_of_time_points": len(dates),
        "first_date": dates[0].isoformat(),
        "last_date": dates[-1].isoformat(),
        "locations": [
            {"id": i + 1, "row": base_row + row_offset, "col": base_col + col_offset}
            for i, (row_offset, col_offset) in enumerate(sample_offsets)
        ],
    }

    # Save metadata to JSON
    import json

    metadata_filename = output_dir / f"displacement_metadata_{timestamp}.json"
    with open(metadata_filename, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_filename}")

    # Optionally create plots
    if args.plot:
        plt.figure(figsize=(10, 6))
        for series in all_series_data:
            location_id = series["location_id"]
            row = series["row"]
            col = series["col"]
            time_series = series["time_series"]

            plt.plot(
                dates,
                time_series,
                marker="o",
                linestyle="-",
                label=f"Loc {location_id}: ({row}, {col})",
            )

        plt.xlabel("Date")
        plt.ylabel("Displacement (m)")
        plt.title("Displacement Time Series")
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        plot_filename = output_dir / f"displacement_timeseries_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300)
        print(f"Plot saved to {plot_filename}")

    print("All processing complete!")


if __name__ == "__main__":
    main()
