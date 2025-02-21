#!/usr/bin/env python
# /// script
# dependencies = ["tyro", "aiohttp"]
# ///
import asyncio
from typing import Any, Literal

import aiohttp
import tyro


async def get_download_url(
    product: dict[str, Any], protocol: Literal["s3", "https"]
) -> str:
    """
    Extract a download URL from the product's UMM metadata.
    Only URLs whose type starts with "GET DATA" and that use the
    requested protocol (https or s3) are returned.
    """
    if protocol not in ["https", "s3"]:
        raise ValueError(f"Unknown protocol {protocol}; must be https or s3")
    for url in product["umm"]["RelatedUrls"]:
        if url["Type"].startswith("GET DATA") and url["URL"].startswith(protocol):
            return url["URL"]
    raise ValueError(f"No download URL found for granule {product['umm']['GranuleUR']}")


async def get_products(
    session: aiohttp.ClientSession,
    frame_id: int,
    product_version: str,
    use_uat: bool = False,
) -> list[dict[str, Any]]:
    """Query the CMR for granules matching the given frame ID and product version."""
    edl_host = "uat.earthdata" if use_uat else "earthdata"
    search_url = f"https://cmr.{edl_host}.nasa.gov/search/granules.umm_json"
    params = {
        "short_name": "OPERA_L3_DISP-S1_V1",
        "attribute[]": [
            f"float,PRODUCT_VERSION,{product_version}",
            f"int,FRAME_NUMBER,{frame_id}",
        ],
        "page_size": 2000,
    }
    headers: dict[str, str] = {}
    products: list[dict[str, Any]] = []
    while True:
        async with session.get(search_url, params=params, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            products.extend(data["items"])
            if "CMR-Search-After" not in response.headers:
                break
            headers["CMR-Search-After"] = response.headers["CMR-Search-After"]
    return products


async def main(
    frame_id: int,
    product_version: str = "1.1",
    protocol: Literal["s3", "https"] = "https",
    use_uat: bool = False,
):
    async with aiohttp.ClientSession() as session:
        products = await get_products(
            session, frame_id, product_version, use_uat=use_uat
        )
        download_tasks = [get_download_url(product, protocol) for product in products]
        urls = await asyncio.gather(*download_tasks)
        for url in urls:
            print(url)


if __name__ == "__main__":
    asyncio.run(tyro.cli(main))
