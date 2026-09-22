# Download the challenge data from the UPM Google Drive link and extract it to the `challenge_data` directory.
# This script is intended to be run once to set up the challenge data for experiments.
import os
import requests
from tqdm import tqdm

def download_and_extract_data():
    # Nextcloud share links serve an HTML page; append "/download" for the raw file
    url = "https://drive.upm.es/s/pzcqnXLnzZSL9Rw/download"
    output_dir = "challenge_data"
    os.makedirs(output_dir, exist_ok=True)

    # Download the file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        zip_path = os.path.join(output_dir, "challenge_data.zip")
        total_size = int(response.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading"
        ) as progress:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                progress.update(len(chunk))

        import zipfile
        if not zipfile.is_zipfile(zip_path):
            os.remove(zip_path)
            raise RuntimeError(
                f"Downloaded file from {url} is not a valid zip file. "
                "The share link may require manual download."
            )
        print(f"Downloaded challenge data to {zip_path}")

        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted challenge data to {output_dir}")

        # Clean up the zip file
        os.remove(zip_path)
    else:
        print(f"Failed to download data. Status code: {response.status_code}")
        
if __name__ == "__main__":
    download_and_extract_data()