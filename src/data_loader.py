
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_raw_data(
    train_path: str | Path | None = None,
    store_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the original Rossmann sales and store datasets."""

    train_path = (
        Path(train_path)
        if train_path is not None
        else RAW_DATA_DIR / "train.csv"
    )
    store_path = (
        Path(store_path)
        if store_path is not None
        else RAW_DATA_DIR / "store.csv"
    )

    if not train_path.exists():
        raise FileNotFoundError(f"Sales file not found: {train_path}")

    if not store_path.exists():
        raise FileNotFoundError(f"Store file not found: {store_path}")

    train = pd.read_csv(train_path, parse_dates=["Date"], dtype={"StateHoliday": "string"},)
    stores = pd.read_csv(store_path)

    return train, stores


def preprocess_store_data(stores: pd.DataFrame) -> pd.DataFrame:
    """Prepare store metadata using the logic from 01_eda.ipynb."""

    stores_clean = stores.copy()

    # Add indicators for missing competition information
    if "CompetitionDistance" in stores_clean.columns:
        stores_clean["CompetitionDistanceMissing"] = (
            stores_clean["CompetitionDistance"]
            .isna()
            .astype("int8")
        )

        # Fill missing competition distances with the median
        median_distance = stores_clean["CompetitionDistance"].median()

        stores_clean["CompetitionDistance"] = (
            stores_clean["CompetitionDistance"]
            .fillna(median_distance)
        )

    competition_date_cols = [
        col
        for col in [
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
        ]
        if col in stores_clean.columns
    ]

    if competition_date_cols:
        stores_clean["CompetitionInfoMissing"] = (
            stores_clean[competition_date_cols]
            .isna()
            .any(axis=1)
            .astype("int8")
        )

    return stores_clean


def merge_sales_and_stores(
    train: pd.DataFrame,
    stores_clean: pd.DataFrame,
) -> pd.DataFrame:
    """Merge sales and store metadata using Store as the key."""

    if train["Store"].isna().any():
        raise ValueError("Missing Store IDs found in sales data.")

    if stores_clean["Store"].isna().any():
        raise ValueError("Missing Store IDs found in store data.")

    if stores_clean["Store"].duplicated().any():
        raise ValueError("Store IDs must be unique in store.csv.")

    missing_store_ids = set(train["Store"]) - set(stores_clean["Store"])

    if missing_store_ids:
        raise ValueError(
            f"Store metadata is missing for IDs: {sorted(missing_store_ids)}"
        )

    df = train.merge(
        stores_clean,
        on="Store",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    # Sprawdzenie poprawności połączenia tabel
    if not (df["_merge"] == "both").all():
        raise ValueError("Some sales rows were not matched to store metadata.")

    df = df.drop(columns="_merge")

    if len(df) != len(train):
        raise ValueError("The merge unexpectedly changed the row count.")

    return df



def load_data(
    train_path: str | Path | None = None,
    store_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load, preprocess and merge the Rossmann datasets."""

    train, stores = load_raw_data(
        train_path=train_path,
        store_path=store_path,
    )

    stores_clean = preprocess_store_data(stores)

    df = merge_sales_and_stores(
        train=train,
        stores_clean=stores_clean,
    )

    return df
