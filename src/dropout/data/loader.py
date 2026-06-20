"""Load and cache the UCI student dropout dataset."""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


UCI_DATASET_ID = 697
DEFAULT_DATA_DIR = Path("data/raw")


def load_dataset(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    force_download: bool = False,
) -> pd.DataFrame:
    """Return the student dropout dataset with a normalized ``target`` column.

    The first call downloads the UCI dataset and stores the feature and target
    tables locally. Later calls reuse those cached CSV files so the notebook and
    training script remain reproducible without repeated network access.
    """
    data_dir = Path(data_dir)
    features_path = data_dir / "features.csv"
    targets_path = data_dir / "targets.csv"

    if not force_download and features_path.exists() and targets_path.exists():
        features = pd.read_csv(features_path)
        targets = pd.read_csv(targets_path)
    else:
        dataset = fetch_ucirepo(id=UCI_DATASET_ID)
        features = dataset.data.features.copy()
        targets = dataset.data.targets.copy()

        data_dir.mkdir(parents=True, exist_ok=True)
        features.to_csv(features_path, index=False)
        targets.to_csv(targets_path, index=False)

    target = targets.iloc[:, 0].rename("target")
    return pd.concat([features, target], axis=1)

