import argparse
import logging
from ersilia import ErsiliaModel
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("Featurization")

def process_dataset(model, input_file, output_file):
    """Processes a single dataset file using the Ersilia model."""
    if not input_file.exists():
        logger.warning(f"Skipping: {input_file} (File not found)")
        return

    logger.info(f"Featurizing: {input_file} -> {output_file}")
    try:
        model.run(input=str(input_file), output=str(output_file))
    except Exception as e:
        logger.error(f"Failed to process {input_file}: {e}")

def main():
    """Main function for dataset featurization."""
    parser = argparse.ArgumentParser(description="Featurize datasets using an Ersilia model")
    parser.add_argument(
        "--model_id",
        type=str,
        default="eos39co",
        help="ID of the Ersilia model to use",
    )
    args = parser.parse_args()

    input_dir = Path("data/download_data")
    output_dir = Path("data/featurized_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Ersilia model: {args.model_id}")
    try:
        model = ErsiliaModel(model=args.model_id)
        model.serve()
    except Exception as e:
        logger.error(f"Failed to load model {args.model_id}: {e}")
        exit(1)

    # Process all CSV files in the input directory
    for input_file in input_dir.glob("*.csv"):
        output_file = output_dir / f"{input_file.stem}_features.csv"
        process_dataset(model, input_file, output_file)

    logger.info("Featurization process completed.")

if __name__ == "__main__":
    main()
