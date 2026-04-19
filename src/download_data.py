"""
download_data.py — Download the online gaming behaviour dataset.

The dataset is hosted on Kaggle. Two download methods are supported:
  1. Kaggle API  (recommended — requires ~/.kaggle/kaggle.json)
  2. Manual URL  (fallback — prints the Kaggle page link)

Usage:
    python -m src.download_data
    python -m src.download_data --output data/
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT = _ROOT / "data"

KAGGLE_DATASET = "rabieelkharoua/predict-online-gaming-behavior-dataset"
EXPECTED_FILENAME = "online_gaming_behavior_dataset.csv"


def download_via_kaggle_api(output_dir: Path) -> Path:
    """Download dataset using the Kaggle Python API.

    Requires the kaggle package and a valid ~/.kaggle/kaggle.json token.

    Parameters
    ----------
    output_dir : Path
        Directory to save the downloaded file.

    Returns
    -------
    Path
        Path to the extracted CSV file.

    Raises
    ------
    ImportError
        If the kaggle package is not installed.
    """
    try:
        import kaggle  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "kaggle package not found. Install with: pip install kaggle"
        ) from e

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset '{KAGGLE_DATASET}' via Kaggle API ...")
    os.system(
        f"kaggle datasets download -d {KAGGLE_DATASET} --unzip -p {output_dir}"
    )

    csv_path = output_dir / EXPECTED_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Download succeeded but expected file not found: {csv_path}"
        )

    print(f"Dataset saved to: {csv_path}")
    return csv_path


def check_data_exists(output_dir: Path) -> bool:
    """Return True if the expected CSV file already exists.

    Parameters
    ----------
    output_dir : Path

    Returns
    -------
    bool
    """
    return (output_dir / EXPECTED_FILENAME).exists()


def main(output_dir=None) -> None:
    """Entry point: download dataset or confirm it already exists.

    Parameters
    ----------
    output_dir : path-like, optional
        Override the default data/ directory.
    """
    out = Path(output_dir) if output_dir else DEFAULT_OUTPUT

    if check_data_exists(out):
        print(f"Dataset already present at: {out / EXPECTED_FILENAME}")
        return

    try:
        download_via_kaggle_api(out)
    except ImportError as exc:
        print(f"\n[WARNING] {exc}")
        print(
            "\nManual download: get the dataset from Kaggle and place it at:\n"
            f"  {out / EXPECTED_FILENAME}\n\n"
            f"Dataset: https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
        )
        sys.exit(1)


def _parse_args():
    p = argparse.ArgumentParser(description="Download the gaming behaviour dataset.")
    p.add_argument("--output", default=None, help="Output directory (default: data/)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(output_dir=args.output)
