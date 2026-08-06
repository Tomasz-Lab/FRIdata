import os
import gzip

import requests

from fridata.utlis.logging import logger

def initialize():
    download_seqres()
    pass


def download_seqres():
    url = "https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
    local_filename = "pdb_seqres.txt.gz"

    # Download the file
    response = requests.get(url, stream=True)
    with open(local_filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)

    logger.info(f"Downloaded {local_filename}")

    # Extract the contents
    with gzip.open(local_filename, "rb") as gz_file:
        extracted_content = gz_file.read()

    extracted_filename = os.path.splitext(local_filename)[0]
    with open(extracted_filename, "wb") as file:
        file.write(extracted_content)

    logger.info(f"Extracted contents to {extracted_filename}")


if __name__ == "__main__":
    initialize()
