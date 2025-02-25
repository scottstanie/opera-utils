import argparse
import time

from opera_utils import _disp
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
        "--row-col",
        type=int,
        nargs=2,
        required=True,
        help="(row, column) within the frame to read from.",
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
    reader = _disp.DispReaderPool(
        urls,
        page_size=args.page_size,
        max_concurrent=args.max_concurrent,
        aws_credentials=aws_credentials,
    )

    # reader = _disp.DispReaderQueue(
    #     urls,
    #     n_workers=args.max_concurrent,
    #     aws_credentials=aws_credentials,
    # )
    # print(f"Created in in {time.time() - t0:.2f} seconds")

    # # Example "requests" to read slices from 2 of the files
    # my_requests = [
    #     (0, (slice(100, 101), slice(None), slice(None))),  # from file 0
    #     (1, (slice(50, ), slice(None), slice(None))),  # from file 1
    # ]
    # results = reader.read_slices(my_requests)
    # # # Now do something with the arrays
    # # # e.g. results might be {1: np.array(...), 2: np.array(...)}
    # # # You can re-assemble them in the order you prefer.
    # # reader.close()

    t0 = time.time()
    reader.open()
    print(f".open() run in {time.time() - t0:.2f} seconds")
    # Example: Read slices of data
    t0 = time.time()
    row, col = args.row_col
    data = reader[:, row : row + 1, col : col + 1]
    print(f"Read data shape {data.shape} in {time.time() - t0:.2f} seconds")

    t0 = time.time()
    data = reader[:, row + 10 : row + 10 + 1, col + 10 : col + 10 + 1]
    print(f"Read data shape {data.shape} in {time.time() - t0:.2f} seconds")

    t0 = time.time()
    data = reader[:, row + 100 : row + 100 + 1, col : col + 1]
    print(f"Read data shape {data.shape} in {time.time() - t0:.2f} seconds")

    reader.close()
    date_strs = [d.strftime("%Y-%m-%d") for d in reader.unique_dates[1:]]
    # pd.DataFrame(data=data)


if __name__ == "__main__":
    main()
