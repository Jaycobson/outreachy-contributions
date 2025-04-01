import os
import sys
import logging
import argparse
import traceback
from pathlib import Path
import pandas as pd
from tdc.single_pred import ADME

# Setup logging
log_file = Path.home() / "dataset_download.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def download_dataset(dataset_name):
    """Download and save the ADME dataset."""
    try:
        output_dir = Path("data/download_data")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading dataset: {dataset_name}")
        adme_data = ADME(name=dataset_name)
        full_dataset = adme_data.get_data()

        # Save full dataset
        full_path = output_dir / f"{dataset_name}_full.csv"
        full_dataset.to_csv(full_path, index=False)
        logger.info(f"Saved full dataset: {full_path}")

        # Save splits
        for split_name, split_data in adme_data.get_split().items():
            split_path = output_dir / f"{dataset_name}_{split_name}.csv"
            pd.DataFrame(split_data).to_csv(split_path, index=False)
            logger.info(f"Saved {split_name} split: {split_path}")

        logger.info("Dataset download completed successfully!")

    except Exception:
        logger.error("Error during dataset download:")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ADME dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the ADME dataset to download (e.g., 'BBB_Martins')"
    )
    args = parser.parse_args()
    download_dataset(args.dataset)
