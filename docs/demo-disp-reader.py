import argparse
import time

from opera_utils._disp import DispReaderPool
from opera_utils.credentials import get_earthaccess_s3_creds


def main():
    """Example usage of the parallel DispReader."""

    parser = argparse.ArgumentParser(description="Read OPERA DISP-S1 files in parallel")
    parser.add_argument(
        "url_file",
        type=str,
        help="File containing S3 URLs to OPERA DISP-S1 files, one per line",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Maximum number of worker processes to use",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=4 * 1024 * 1024,
        help="Page size in bytes for HDF5 file system page strategy",
    )
    args = parser.parse_args()

    # Read URLs from file
    with open(args.url_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Found {len(urls)} URLs to process")

    aws_credentials = get_earthaccess_s3_creds("opera-uat")

    # Create and use the reader
    t0 = time.time()
    reader = DispReaderPool(
        urls,
        page_size=args.page_size,
        max_concurrent=args.max_concurrent,
        aws_credentials=aws_credentials,
    )
    print(f"Created in in {time.time() - t0:.2f} seconds")

    t0 = time.time()
    reader.open()
    print(f".open() run in {time.time() - t0:.2f} seconds")
    # Example: Read slices of data
    t0 = time.time()
    data = reader[:, 100:200, 100:200]
    print(f"Read data shape {data.shape} in {time.time() - t0:.2f} seconds")

    t0 = time.time()
    data = reader[:, 1000:1100, 1100:1200]
    print(f"Read data shape {data.shape} in {time.time() - t0:.2f} seconds")

    t0 = time.time()
    data = reader[:, 5000:5100, 5100:5200]
    print(f"Read data shape {data.shape} in {time.time() - t0:.2f} seconds")
    reader.close()


if __name__ == "__main__":
    main()
